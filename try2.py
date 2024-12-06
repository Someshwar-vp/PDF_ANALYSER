# Import necessary libraries
import fitz  # PyMuPDF (for extracting text from PDF files)
import os
from uuid import uuid4  # To generate unique IDs for documents

# FAISS and Sentence Transformers imports
import faiss  # FAISS (for similarity search and vector storage)
from sentence_transformers import SentenceTransformer  # For generating embeddings from sentences
from langchain_community.vectorstores import FAISS  # LangChain's FAISS vector store
from langchain_community.docstore.in_memory import InMemoryDocstore  # In-memory docstore to hold documents
from langchain_core.documents import Document  # To represent documents in LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter  # To split large texts into smaller chunks
from langchain import hub  # To fetch pre-built templates
from langchain_groq import ChatGroq  # For integrating with ChatGroq language model
from langchain_community.embeddings import SentenceTransformerEmbeddings  # Embedding class for SentenceTransformer

# Set environment variables for API keys (used for interacting with third-party APIs like Groq)
os.environ["GROQ_API_KEY"] = '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'
os.environ["LANGCHAIN_API_KEY"] = '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'

# Initialize the SentenceTransformer model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # The model used to convert text to embeddings (a numerical representation)

# Wrap the model name with LangChain's Embedding class
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the HuggingFaceEmbeddings with the model name
embedding = HuggingFaceEmbeddings(model_name=model_name)  # The model is now wrapped in LangChain's HuggingFaceEmbeddings class

# Initialize FAISS vector store
index = faiss.IndexFlatL2(384)  # 384 is the dimension size of embeddings generated by 'all-MiniLM-L6-v2' model
# FAISS is used for storing and searching vectorized documents (embeddings)
vector_store = FAISS(
    embedding_function=embedding,  # Embedding function (used to generate vectors for documents)
    index=index,  # FAISS index to store and search vectors
    docstore=InMemoryDocstore(),  # In-memory docstore to hold documents alongside their vectors
    index_to_docstore_id={},  # A mapping between the index and docstore (for fast retrieval)
)

# Initialize the ChatGroq language model (used for generating responses based on queries)
llm = ChatGroq(model="llama3-8b-8192")  # Specify the model to use for generating responses

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""  # Initialize an empty string to store the extracted text
    # Open the PDF file
    with fitz.open(pdf_path) as pdf:
        # Iterate over each page of the PDF
        for page in pdf:
            text += page.get_text()  # Extract text from the page and append it to 'text'
    return text  # Return the extracted text

# Function to analyze the PDF and answer a specific question
def analyze_pdf(pdf_path, question):
    # Extract text from the PDF using the above function
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Create a Document object from the extracted PDF text
    doc = Document(page_content=pdf_text)

    # Split the document text into smaller chunks (for easier processing and better retrieval)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Split text into 1000-character chunks with 200-character overlap
    all_splits = text_splitter.split_documents([doc])  # Split the document into chunks

    # Generate unique IDs for each chunk and add the chunks to the FAISS vector store
    uuids = [str(uuid4()) for _ in range(len(all_splits))]  # Generate unique IDs
    vector_store.add_documents(documents=all_splits, ids=uuids)  # Add the document chunks to the vector store

    # Fetch a pre-built prompt template for question-answering (from LangChain's Hub)
    prompt = hub.pull("rlm/rag-prompt")

    # Perform a similarity search to find the most relevant document chunks for the given question
    retrieved_docs = vector_store.similarity_search(question, k=5)  # Retrieve top 5 most relevant chunks

    # Combine the content of the retrieved documents to form the context for answering the question
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Invoke the question-answering system with the question and context
    messages = prompt.invoke({"question": question, "context": docs_content})
    
    # Generate a response from the language model based on the question and context
    response = llm.invoke(messages)

    return response.content  # Return the generated response

# Example usage (analyzing a specific PDF and asking a question)
pdf_path = "NCCN_breastcancer.pdf"  # Specify the PDF file path
question = "A 67-year-old woman with a history of ER+/HER2-negative breast cancer treated 5 years ago now presents with recurrent disease in the bone. What systemic therapy options should be considered?"  # Define the question to ask
answer = analyze_pdf(pdf_path, question)  # Analyze the PDF and get the answer
print(answer)  # Print the answer







