
# Medical Chatbot using Llama 2 🩺🤖

## Introduction
Welcome to the Medical Chatbot project, powered by Llama 2, Langchain, and Chainlit. This chatbot leverages advanced language models and vector stores to provide accurate responses to medical queries.

## Project Overview

### Data Preparation
We used the "Gale Encyclopedia of Medicine Book Volume 1" to train our model. The data preprocessing involved text splitting using RecursiveCharacterTextSplitter and embeddings with Hugging Face.

### Vector Database Creation
The vector database was created using the FAISS vector model with Langchain. The `ingest.py` script processes PDF documents, splits them into texts, and creates a vector database for efficient retrieval.

```bash
python ingest.py
```

### Model Deployment
Our chatbot uses the Llama 2 language model (`llama-2-7b-chat.ggmlv3.q8_0.bin`) and is deployed as a web application and chatbot using Chainlit.

```bash
python model.py
```

## Code Modules

### Ingest.py
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

#create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device':'cuda'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == '__main__':
    create_vector_db()



### Model.py
```python
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "D:\End to end project tutorial\AI for everyone\Projects\Llama2-Medical-Chatbot\llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,     #Hyperparameters
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start    #decorators
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()



## Usage

### Setting Up the Environment
Make sure to install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Chatbot
Execute the `model.py` script to start the chatbot:

```bash
python model.py
```

## Output Screenshots

![Screenshot 1](https://github.com/HarishNandhan/Medical_chatbot_using_LLAMA2/blob/main/data/welcomepage.jpg)

![Screenshot 2](https://github.com/HarishNandhan/Medical_chatbot_using_LLAMA2/blob/main/data/prompt%20response.jpg)

Feel free to contribute and improve this Medical Chatbot project! If you encounter any issues, please open an [issue](https://github.com/yourusername/medical-chatbot/issues).
