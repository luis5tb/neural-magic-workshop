# Model Quantization Pipeline

This pipeline automates the process of quantizing machine learning models. It handles downloading the model from Hugging Face, quantizing it, uploading the quantized version, and evaluating its accuracy.

## Pipeline Overview

The pipeline consists of the following stages:

1. **Create PVC**: Creates a Persistent Volume Claim for storing model data
2. **Download Model**: Downloads the specified model from Hugging Face Hub
3. **Quantize Model**: Performs model quantization (supports int8 quantization)
4. **Upload Model**: Uploads the quantized model to a storage location
5. **Evaluate Model**: Evaluates the quantized model's accuracy
6. **Delete PVC**: Cleans up by deleting the PVC after completion

## Prerequisites

- Python 3.12
- Kubeflow Pipelines SDK (`kfp`)
- Access to OpenShift AI (formerly Red Hat OpenShift Data Science)
- S3-compatible storage data connection configured in OpenShift AI

## Workbench Installation Instructions

Install the required dependencies:

```bash
pip install kfp
```

## Compiling the Pipeline

To compile the pipeline into a YAML file that can be imported into OpenShift AI:

```bash
python quantization_pipeline.py
```

This will generate a `quantization_pipeline.yaml` file. Download it to your local machine.

## Using the Pipeline

1. Log into your OpenShift AI instance
2. Navigate to "Data Science Pipelines" → "Pipelines"
3. Click "Import Pipeline"
4. Choose "Upload file" and select the generated `quantization_pipeline.yaml`
5. Click "Import pipeline"

### Pipeline Parameters

When running the pipeline, you can configure the following parameters:

- `model_id`: The Hugging Face model ID (default: "ibm-granite/granite-3.2-2b-instruct")
- `output_path`: Path for the quantized model (default: "granite-int8")
- `quantization_type`: Type of quantization to perform (default: "int8")
- `data_connection`: Data connection name for model storage (default: "aws-connection-models")

## Storage Requirements

The pipeline creates a PVC with:
- Size: 30Gi
- Access Mode: ReadWriteMany
- Storage Class: standard

Make sure your cluster has the appropriate storage class available.

## Data Connection Setup

Before running the pipeline:

1. Create a data connection in OpenShift AI pointing to your S3 storage, named "pipeline-s3-connection"
2. The data connection has the next mandatory fields:
   - Connection name: pipeline-s3-connection
   - Access Key
   - Secret Key
   - Endpoint
   - Bucket: need to ensure the bucket exists on S3 (minio) before triggering the pipeline
