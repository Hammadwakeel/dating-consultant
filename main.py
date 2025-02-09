from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
from dotenv import load_dotenv
import os
import uuid
import asyncio
import tempfile
import shutil
import zipfile  # For handling ZIP files
from urllib.parse import quote_plus

from starlette.background import BackgroundTask

# Load environment variables
load_dotenv()

# Import necessary modules from the project
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain

# Initialize FastAPI app
app = FastAPI()

# Models
class DocumentIDs(BaseModel):
    ids: List[str]

class ChatQuery(BaseModel):
    chat_id: str
    query: str

# Initialize components
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PASSWORD = os.getenv("MONGO_PASSWORD")

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

dimensions: int = len(embeddings.embed_query("dummy"))

vectorstore = FAISS(
    embedding_function=embeddings,
    index=IndexFlatL2(dimensions),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    normalize_L2=False
)

# MongoDB connection details
password = quote_plus(os.getenv("MONGO_PASSWORD"))
database_name = os.getenv("DATABASE_NAME")
collection_name = os.getenv("COLLECTION_NAME")
connection_string = os.getenv("CONNECTION_STRING").replace("${PASSWORD}", password)
chat_sessions = {}

system_prompt = '''
You are a helpful assistant, with a compassionate and supportive approach, similar to that of a dating consultant or relationship advisor. Your goal is to provide insightful, practical, and considerate advice on relationships, dating, marriage, and personal growth. You should offer suggestions and guidance in a friendly, empathetic tone, aiming to help the user navigate their questions or concerns.

Guidelines:
1. Approach the user's query with understanding and empathy.
2. Offer thoughtful and actionable advice, drawing from relationship principles and common wisdom.
3. Be respectful and non-judgmental in all responses, always seeking to empower the user with helpful perspectives.
4. If the question requires more information or clarification, ask politely and gently.

Retrieved context:
{context}

User's question:
{question}

Your response:
'''

# Helper methods
def create_new_chat():
    """
    Creates a new chat session by initializing a MongoDBChatMessageHistory object.
    Returns:
        str: The unique chat ID.
    """
    chat_id = str(uuid.uuid4())
    chat_message_history = MongoDBChatMessageHistory(
        session_id=chat_id,
        connection_string=connection_string,
        database_name=database_name,
        collection_name=collection_name,
    )
    chat_sessions[chat_id] = chat_message_history
    return chat_id

def get_chat_history(chat_id):
    return chat_sessions.get(chat_id)

def summarize_messages(chat_history):
    stored_messages = chat_history.messages
    if not stored_messages:
        return False

    summarization_prompt = ChatPromptTemplate.from_messages([
        ("placeholder", "{chat_history}"),
        ("human", "Summarize the above chat messages into a single concise message. Include as many specific details as you can."),
    ])

    summarization_chain = summarization_prompt | embeddings
    summary_message = summarization_chain.invoke({"chat_history": stored_messages})

    chat_history.clear()
    chat_history.add_ai_message(summary_message.content)
    return True

def stream_chat_response(query, chat_id):
    chat_history = get_chat_history(chat_id)
    if not chat_history:
        raise HTTPException(status_code=404, detail="Chat history not found")

    summarize_messages(chat_history)

    input_data_for_chain = {
        "question": query,
        "chat_history": chat_history
    }

    chain = ConversationalRetrievalChain.from_llm(
        llm=embeddings,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type='stuff',
        combine_docs_chain_kwargs={"prompt": system_prompt},
        verbose=False
    )

    response_stream = chain.stream(input_data_for_chain)

    final_answer = ""
    for chunk in response_stream:
        if 'answer' in chunk:
            print(chunk['answer'], end="", flush=True)
            final_answer = chunk['answer']
        else:
            print(f"Unexpected chunk structure: {chunk}", flush=True)

    if final_answer:
        chat_history.add_ai_message(final_answer)
    else:
        print(f"Unable to extract final content from the last chunk: {chunk}", flush=True)
    return final_answer

def cleanup_files(original: str, served: Optional[str] = None):
    """
    Deletes the saved vectorstore files from disk.
    If `original` is a directory, it removes it.
    If a separate served file exists (like a zip file), it is removed as well.
    """
    try:
        if os.path.exists(original):
            if os.path.isdir(original):
                shutil.rmtree(original)
            else:
                os.remove(original)
        if served and served != original and os.path.exists(served):
            os.remove(served)
    except Exception as e:
        print(f"Error cleaning up files: {e}")

# Endpoints
@app.post("/add-document")
async def add_document(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(file_path=tmp_path, extract_images=True)
        docs = list(loader.lazy_load())
        ids = [str(uuid4()) for _ in range(len(docs))]
        vectorstore.add_documents(documents=docs, ids=ids, embeddings=embeddings)
    finally:
        os.remove(tmp_path)

    return {"message": "Documents added successfully", "ids": ids}

@app.delete("/delete-document")
async def delete_document(doc_ids: DocumentIDs):
    success = vectorstore.delete(ids=doc_ids.ids)
    if success:
        return {"message": "Documents deleted successfully"}
    raise HTTPException(status_code=400, detail="Failed to delete documents")

@app.post("/save-download-vectorstore")
async def save_download_vectorstore(filename: str, background_tasks: BackgroundTasks):
    """
    Save the current vectorstore to disk and immediately return it as a downloadable file.
    After the file is served, the saved vectorstore (and any created zip file) is deleted.
    """
    # Save the vectorstore locally
    vectorstore.save_local(filename)

    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Saved vectorstore not found.")

    # If the saved vectorstore is a directory, compress it into a zip file.
    if os.path.isdir(filename):
        zip_filename = filename + ".zip"
        shutil.make_archive(filename, 'zip', filename)
        file_to_serve = zip_filename
        media_type = "application/zip"
        serve_filename = os.path.basename(zip_filename)
        original = filename
        served = zip_filename
    else:
        file_to_serve = filename
        media_type = "application/octet-stream"
        serve_filename = os.path.basename(filename)
        original = filename
        served = None

    # Schedule deletion of the saved files after the response is sent.
    bg_task = BackgroundTask(cleanup_files, original, served)
    return FileResponse(
        path=file_to_serve,
        filename=serve_filename,
        media_type=media_type,
        background=bg_task
    )

@app.post("/load-vectorstore")
async def load_vectorstore(file: UploadFile = File(...)):
    """
    Load a vectorstore from an uploaded file. If the uploaded file is a ZIP file,
    it is extracted to a temporary directory before loading.
    """
    global vectorstore  # Declare global variable at the top
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_filename = tmp.name

    try:
        if zipfile.is_zipfile(tmp_filename):
            with tempfile.TemporaryDirectory() as extract_dir:
                with zipfile.ZipFile(tmp_filename, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                extracted_items = os.listdir(extract_dir)
                if len(extracted_items) == 1:
                    potential_dir = os.path.join(extract_dir, extracted_items[0])
                    if os.path.isdir(potential_dir):
                        vectorstore_dir = potential_dir
                    else:
                        vectorstore_dir = extract_dir
                else:
                    vectorstore_dir = extract_dir

                vectorstore = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)
            message = "Vector store loaded successfully from ZIP."
        else:
            vectorstore = FAISS.load_local(tmp_filename, embeddings, allow_dangerous_deserialization=True)
            message = "Vector store loaded successfully."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading vectorstore: {str(e)}")
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
    return {"message": message}

@app.post("/merge-vectorstore")
async def merge_vectorstore(file: UploadFile = File(...)):
    """
    Merge an uploaded vectorstore into the current vectorstore.
    If the uploaded file is a ZIP file, it is extracted to a temporary directory before loading.
    """
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_filename = tmp.name

    try:
        # Check if the uploaded file is a ZIP archive
        if zipfile.is_zipfile(tmp_filename):
            with tempfile.TemporaryDirectory() as extract_dir:
                with zipfile.ZipFile(tmp_filename, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                extracted_items = os.listdir(extract_dir)
                if len(extracted_items) == 1:
                    potential_dir = os.path.join(extract_dir, extracted_items[0])
                    if os.path.isdir(potential_dir):
                        vectorstore_dir = potential_dir
                    else:
                        vectorstore_dir = extract_dir
                else:
                    vectorstore_dir = extract_dir

                source_store = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)
        else:
            source_store = FAISS.load_local(tmp_filename, embeddings, allow_dangerous_deserialization=True)
        vectorstore.merge_from(source_store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error merging vectorstore: {str(e)}")
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
    return {"message": "Vector stores merged successfully"}

@app.post("/create-chat")
async def create_chat():
    chat_id = create_new_chat()
    return {"chat_id": chat_id, "message": "New chat created successfully"}

@app.post("/generate-answer")
async def generate_answer(chat_query: ChatQuery):
    chat_id = chat_query.chat_id
    query = chat_query.query

    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat ID not found")

    response = stream_chat_response(query, chat_id)
    return {"answer": response}

@app.get("/")
async def root():
    return {"message": "API is up and running!"}

def start_app():
    print("FastAPI application is starting...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
