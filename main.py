from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
import pinecone
from dotenv import load_dotenv
import os

class ChatBot():
  load_dotenv()
  loader = TextLoader('./Dep3.txt')
  documents = loader.load()
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
  docs = text_splitter.split_documents(documents)

  embeddings = HuggingFaceEmbeddings()
  #print(os.getenv('PINECONE_API_KEY'))
  pinecone.init(
      api_key= "a38d4e80-cd1b-465b-82ce-077b9622b58c",
      environment='gcp-starter'
  )

  index_name = "langchain-demo"

  if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric="cosine", dimension=768)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
  else:
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

  repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
  llm = HuggingFaceHub(
      repo_id=repo_id, model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50}, huggingfacehub_api_token="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"
  )

  from langchain import PromptTemplate

  template = """
  You are a symptom tracking chatbot with the goal of talking with someone and returning a list of symptoms and their severity for the mental illness depression. 
  If you are given any questions not related to mental illness or medicine, answer with a variation of \"Sorry, i cannot answer that\".
  If they answer with a variation of \"Thank you, im done\", then return the list of symptoms with their severities in the way that a doctor might use it
  Use the documents provided to ask questions to get the possible symptoms that the user has without actually referring to depression
  The user will initiate the conversation, but you respond by asking questions.
  You answer with short and concise answer, no longer than 2 sentences.

  past messages: {pasts}

  Context: {context}
  Question: {question}
  Answer: 

  """

  prompt = PromptTemplate(template=template, input_variables=["context", "question","pasts"])

  from langchain.schema.runnable import RunnablePassthrough
  from langchain.schema.output_parser import StrOutputParser

  """ rag_chain = (
    {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
  ) """
  rag_chain = (
    prompt 
    | llm
    | StrOutputParser() 
  )
