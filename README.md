FastAPI Vectorstore & Conversational AI API

This project implements a RESTful API using FastAPI to manage a FAISS-based vectorstore and power a conversational AI assistant. It leverages LangChain for chaining tasks, OpenAI embeddings for semantic search, and MongoDB to store chat message histories. The API provides endpoints to upload documents, manage the vectorstore (save, load, merge), and interact with a chat interface for generating AI responses based on context retrieved from the vectorstore.
Features

    Document Upload & Indexing
    Upload PDF documents, extract text and images via a PDF loader, and index them in a FAISS vectorstore.

    Vectorstore Management
        Save & Download: Save the current vectorstore and download it as a ZIP archive (with automatic cleanup).
        Load: Load an existing vectorstore from a file (supports ZIP uploads).
        Merge: Merge an uploaded vectorstore with the current one (supports ZIP uploads).

    Conversational Chatbot Interface
        Chat Sessions: Create new chat sessions that store conversation history in MongoDB.
        Generate Answer: Generate chatbot responses using a conversational retrieval chain that combines context from the vectorstore with a dynamic prompt.

    Automated Cleanup
    Temporary files are cleaned up automatically after file operations to keep the environment tidy.

Prerequisites

    Python 3.12 (or a compatible version)
    FastAPI
    Uvicorn
    python-dotenv
    FAISS
    LangChain-related libraries:
        langchain_community.vectorstores
        langchain_openai
        langchain_mongodb.chat_message_histories
        langchain_community.document_loaders
        langchain.prompts
        langchain.chains
    Other packages such as pydantic

Install dependencies (you might use a requirements.txt file):

pip install fastapi uvicorn python-dotenv pydantic faiss-cpu
# Also install any additional LangChain community libraries required.

Installation

    Clone the Repository:

git clone https://github.com/yourusername/yourproject.git
cd yourproject

Set Up a Virtual Environment:

python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate

Install Required Packages:

    pip install -r requirements.txt

Environment Variables

Create a .env file in the project root with the following contents (adjust the values as needed):

OPENAI_API_KEY=your_openai_api_key
MONGO_PASSWORD=your_mongo_password
CONNECTION_STRING=your_mongodb_connection_string
DATABASE_NAME=your_database_name
COLLECTION_NAME=your_collection_name

Running the Application

Start the FastAPI server with:

uvicorn main:app --reload

The API will be available at http://127.0.0.1:8000/.
API Endpoints
Document Management

    Add Document
    Endpoint: POST /add-document
    Description: Upload a PDF file to extract and index its contents.
    Parameters: PDF file via form data.

    Delete Document
    Endpoint: DELETE /delete-document
    Description: Delete documents from the vectorstore using their IDs.
    Example Request Body:

    {
      "ids": ["document_id_1", "document_id_2"]
    }

    Save & Download Vectorstore
    Endpoint: POST /save-download-vectorstore
    Description: Save the current vectorstore to disk and immediately return it as a downloadable ZIP file. The saved files are automatically deleted after download.
    Parameters: filename (e.g., "vectorstore_data") as a query parameter or form field.

Vectorstore Management

    Load Vectorstore
    Endpoint: POST /load-vectorstore
    Description: Load a vectorstore from an uploaded file. Supports both raw files and ZIP archives.

    Merge Vectorstore
    Endpoint: POST /merge-vectorstore
    Description: Merge an uploaded vectorstore with the current vectorstore. Supports ZIP uploads.

Conversational Chatbot

    Create Chat Session
    Endpoint: POST /create-chat
    Description: Create a new chat session with a unique ID and initialize the MongoDB chat history.

    Generate Answer
    Endpoint: POST /generate-answer
    Description: Generate a chatbot response based on a query and the chat history.
    Example Request Body:

    {
      "chat_id": "your_chat_session_id",
      "query": "How can I improve my communication skills in a relationship?"
    }

    Root Endpoint
    Endpoint: GET /
    Description: Health-check endpoint that confirms the API is running.

Contributing

Contributions are welcome! Please open issues or submit pull requests if you have improvements or bug fixes.
