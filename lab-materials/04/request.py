from langchain_openai import ChatOpenAI
import httpx

MODEL = YOUR_MODLE_DEPLOYMENT_NAME
URL = YOUR_MODEL_PUBLIC_ENDPOINT
API_KEY = YOUR_MODEL_TOKEN

llm = ChatOpenAI(
   model=MODEL,
   temperature=0,
   timeout=None,
   max_retries=2,
   api_key=API_KEY,
   base_url=f"{URL}/v1",
   http_client=httpx.Client(verify=False)
)

messages = [
   (
       "system",
       "You are a helpful assistant.",
   ),
   ("human", "What is OpenShift?"),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
