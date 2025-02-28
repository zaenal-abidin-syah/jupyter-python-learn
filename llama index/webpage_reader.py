from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
import llama_index
# from llama_index.llms.openai import OpenAI
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv
from llama_index.core import Settings
api_key = os.getenv("OPEN_API_KEY")
# print(api_key)
# my_model = OpenAILike(model="gpt-4o-mini", api_base="https://free-ddc.xiolabs.xyz/v1", api_key=api_key)

model = OpenAI(
  model="gpt-3.5-turbo",
  api_base="https://free-ddc.xiolabs.xyz/v1",
  api_key=api_key,
)
from llama_index.core.llms import ChatMessage

gen = model.chat([ChatMessage(role="user", content="Hello")])
print(gen)


# Settings.embed_model = OpenAILike(
#   model="models/text-embedding-ada-002",
#   api_base="https://free-ddc.xiolabs.xyz/v1",
#   api_key=api_key
# )

# print(Settings.embed_model)
# Settings.embed_model = HuggingFaceEmbedding(
#     # model_name="v2ray/GPT4chan-8B-QLoRA"
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
#     # model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
#     # model_name="Xenova/text-embedding-ada-002"
#     # model_name="google/flan-t5-large"
#   )
# Settings.llm = None

# load_dotenv()
# # openai_obj1 = OpenAI(api_base='https://api-handler-ddc-free-api.hf.space/v2', api_key=api_key)

# def  main(url: str):
#   # embed_model = HuggingFaceEmbedding(
#   #   model_name="v2ray/GPT4chan-8B-QLoRA"
#   #   # model_name="sentence-transformers/all-MiniLM-L6-v2"
#   #   # model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
#   #   # model_name="Xenova/text-embedding-ada-002"
#   # )
#   response = Settings.llm.complete("halo")
#   print(response)
#   # Settings.llm.ch('jelaskan tentang machine learning')
  
#   # document = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
#   # index = VectorStoreIndex.from_documents(documents=document, embed_model=Settings.embed_model)
#   # # index = VectorStoreIndex.from_documents(documents=document, embed_model=embed_model)
#   # query_engine = index.as_query_engine(llm=Settings.llm)
#   # response = query_engine.query("what is Machine Learning?")
#   # print(type(response))
#   # print(response)

# if __name__ == "__main__":
#   main(url="https://medium.com/edureka/machine-learning-tutorial-f2883412fba1")
#   # v2ray/GPT4chan-8B-QLoRA