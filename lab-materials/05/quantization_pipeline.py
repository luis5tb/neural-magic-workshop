from kfp import compiler, dsl
from kfp.kubernetes import (
    add_toleration,    
    CreatePVC,
    DeletePVC,
    mount_pvc,
    use_secret_as_env,
)

MOUNT_POINT = "/models"
BASE_MODEL_PATH = MOUNT_POINT + "/base-model"
OPTIMIZED_MODEL_PATH = MOUNT_POINT + "/optimized-model"


@dsl.component(
    base_image='registry.access.redhat.com/ubi9/python-312',
    packages_to_install=[
        'huggingface-hub',
        #'boto3',
    ]
)
def download_model(
    model_id: str,
    output_path: str,
):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_id, local_dir=output_path)
    print("Model downloaded successfully from HF.")


@dsl.component(
    base_image='registry.access.redhat.com/ubi9/python-312',
    packages_to_install=[
        'llmcompressor',
        'accelerate',
        'vllm'
    ]
)
def quantize_model(
    model_path: str,
    output_path: str,
    quantization_type: str,
):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # 1) Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2) Data calibration
    from datasets import load_dataset

    # Exercise left for the attendance:
    # This is harcoded but it could be parametrized in the pipeline
    NUM_CALIBRATION_SAMPLES = 256  # 1024
    DATASET_ID = "neuralmagic/LLM_compression_calibration"
    DATASET_SPLIT = "train"

    # Load dataset.
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        return {"text": example["text"]}
    ds = ds.map(preprocess)

    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # 3) Quantize model
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.transformers import oneshot
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    # Exercise left for the attendance:
    # This is harcoded but it could be parametrized in the pipeline
    DAMPENING_FRAC = 0.1  # 0.01
    OBSERVER = "mse"  # minmax
    GROUP_SIZE = 128  # 64
    # Configure the quantization algorithm to run.
    ignore=["lm_head"]
    mappings=[
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
        [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
        [["re:.*down_proj"], "re:.*up_proj"]
    ]
    if quantization_type == "int8":
        recipe = [
            SmoothQuantModifier(smoothing_strength=0.7, ignore=ignore, mappings=mappings),
            GPTQModifier(
                targets=["Linear"],
                ignore=ignore,
                scheme="W8A8",
                dampening_frac=DAMPENING_FRAC,
                observer=OBSERVER,
            )
        ]
    elif quantization_type == "int4":
        recipe = [
            GPTQModifier(
                targets=["Linear"],
                ignore=ignore,
                scheme="w4a16",
                dampening_frac=DAMPENING_FRAC,
                observer=OBSERVER,
                group_size=GROUP_SIZE
            )
        ]
    else:
        raise ValueError(f"Quantization type {quantization_type} not supported")

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        max_seq_length=8196,
    )

    # Save to disk compressed.
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

@dsl.component(
    base_image='registry.access.redhat.com/ubi9/python-312',
    packages_to_install=[
        'boto3',
    ]
)
def upload_model(
    model_path: str,
    s3_path: str,
):
    import os
    from boto3 import client

    print('Starting results upload.')
    s3_endpoint_url = os.environ["s3_host"]
    s3_access_key = os.environ["s3_access_key"]
    s3_secret_key = os.environ["s3_secret_access_key"]
    s3_bucket_name = os.environ["s3_bucket"]

    print(f'Uploading predictions to bucket {s3_bucket_name} '
          f'to S3 storage at {s3_endpoint_url}')

    s3_client = client(
        's3', endpoint_url=s3_endpoint_url, aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key, verify=False
    )

    # Walk through the local folder and upload files
    for root, dirs, files in os.walk(model_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_file_path = os.path.join(s3_path, local_file_path[len(model_path)+1:])
            s3_client.upload_file(local_file_path, s3_bucket_name, s3_file_path)
            print(f'Uploaded {local_file_path}')

    print('Finished uploading results.')


@dsl.component(
    base_image='registry.access.redhat.com/ubi9/python-312',
    packages_to_install=[
        'lm_eval==v0.4.3',
        'vllm',
    ]
)
def evaluate_model(
    model_path: str,
):
    """ Command to execute:
    lm_eval --model vllm \
      --model_args pretrained=$MODEL_PATH,add_bos_token=true \
      --trust_remote_code \
      --tasks gsm8k \
      --num_fewshot 5 \
      --limit 250 \
      --batch_size 'auto'
    """
    import subprocess
    import os

    model_args = "pretrained=" + model_path  + ",add_bos_token=true"

    # Execute the huggingface_hub-cli command
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(["lm_eval",
                             "--model", "vllm",
                             "--model_args", model_args,
                             "--trust_remote_code",
                             "--tasks", "gsm8k",
                             "--num_fewshot", "5",
                             "--limit", "250",
                             "--batch_size", "auto"],
                            capture_output=True, text=True, env=env)
    # Check for errors or output
    if result.returncode == 0:
        print("Model evaluated successfully:")
        print(result.stdout)
    else:
        print("Error evaluating the model:")
        print(result.stderr)


@dsl.pipeline(
    name="Quantization Pipeline",
    description="A pipeline for quantizing a model"
)
def quantization_pipeline(
    model_id: str="ibm-granite/granite-3.2-2b-instruct",
    output_path: str="granite-int8-pipeline",
    quantization_type: str="int8",
):
    #Steps:
    # 1) Download model
    # 2) Quantize model
    # 3) Upload model
    # 4) Evaluate model

    storage_class = "gp3-csi"
    secret_name = "minio-models"
    # Pipeline stage
    quantization_pvc_task = CreatePVC(
        pvc_name_suffix="-quantization",
        access_modes=["ReadWriteOnce"],
        size="30Gi",
        storage_class_name=storage_class,
    )

    download_model_task = download_model(
        model_id=model_id,
        output_path=BASE_MODEL_PATH,
    )
    download_model_task.set_caching_options(False)
    # GPU is not needed, but added to ensure the PVC is associated to a region with GPUs
    # This avoids the following tasks are not in pending status due to either having GPUs or PVC affinity
    download_model_task.set_accelerator_limit(1)
    download_model_task.set_accelerator_type("nvidia.com/gpu")
    add_toleration(download_model_task,
                   key='nvidia.com/gpu',
                   operator='Exists',
                   effect='NoSchedule')
    mount_pvc(
        task=download_model_task,
        pvc_name=quantization_pvc_task.output,
        mount_path=MOUNT_POINT,
    )
    
    # TODO: Only int8 and int4 supported. Add support for fp8 type
    quantize_model_task = quantize_model(
        model_path=BASE_MODEL_PATH,
        output_path=OPTIMIZED_MODEL_PATH,
        quantization_type=quantization_type,
    )
    quantize_model_task.set_caching_options(False)
    quantize_model_task.after(
        download_model_task,
    )
    quantize_model_task.set_accelerator_limit(1)
    quantize_model_task.set_accelerator_type("nvidia.com/gpu")
    add_toleration(quantize_model_task,
                   key='nvidia.com/gpu',
                   operator='Exists',
                   effect='NoSchedule')
    mount_pvc(
        task=quantize_model_task,
        pvc_name=quantization_pvc_task.output,
        mount_path=MOUNT_POINT,
    )

    upload_model_task = upload_model(
        model_path=OPTIMIZED_MODEL_PATH,
        s3_path=output_path,
    )
    upload_model_task.set_caching_options(False)
    upload_model_task.after(
        quantize_model_task,
    )
    mount_pvc(
        task=upload_model_task,
        pvc_name=quantization_pvc_task.output,
        mount_path=MOUNT_POINT,
    )
    use_secret_as_env(upload_model_task,
                      secret_name=secret_name,
                      secret_key_to_env={'AWS_ACCESS_KEY_ID': 's3_access_key',
                                         'AWS_SECRET_ACCESS_KEY': 's3_secret_access_key',
                                         'AWS_S3_ENDPOINT': 's3_host',
                                         'AWS_S3_BUCKET':'s3_bucket'})

    evaluate_model_task = evaluate_model(
        model_path=OPTIMIZED_MODEL_PATH,
    )
    evaluate_model_task.set_caching_options(False)
    evaluate_model_task.after(
        quantize_model_task,
    )
    evaluate_model_task.set_accelerator_limit(1)
    evaluate_model_task.set_accelerator_type("nvidia.com/gpu")
    add_toleration(evaluate_model_task,
                   key='nvidia.com/gpu',
                   operator='Exists',
                   effect='NoSchedule')
    mount_pvc(
        task=evaluate_model_task,
        pvc_name=quantization_pvc_task.output,
        mount_path=MOUNT_POINT,
    )

    quantization_pvc_delete_task = DeletePVC(
        pvc_name=quantization_pvc_task.output,
    )
    quantization_pvc_delete_task.after(
        upload_model_task,
        evaluate_model_task,
    )


compiler.Compiler().compile(quantization_pipeline, package_path='quantization_pipeline.yaml')