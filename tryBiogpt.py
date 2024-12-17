# Import necessary libraries
import streamlit as st
import fitz  # PyMuPDF (for extracting text from PDF files)
import os  # For interacting with the operating system (e.g., setting environment variables)
from uuid import uuid4  # To generate unique IDs for documents
import faiss  # FAISS (for similarity search and vector storage)
from sentence_transformers import SentenceTransformer  # For generating embeddings from sentences
from langchain_community.vectorstores import FAISS  # LangChain's FAISS vector store
from langchain_community.docstore.in_memory import InMemoryDocstore  # In-memory docstore to hold documents
from langchain_core.documents import Document  # To represent documents in LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter  # To split large texts into smaller chunks
from langchain import hub  # To fetch pre-built templates
from langchain_community.embeddings import SentenceTransformerEmbeddings  # Embedding class for SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM  # HuggingFace for BioGPT model

# Initialize BioGPT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMedQA")

# Initialize SentenceTransformer embeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embedding = SentenceTransformerEmbeddings(model_name=model_name)

# Initialize FAISS vector store
index = faiss.IndexFlatL2(384)  
vector_store = FAISS(
    embedding_function=embedding,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Function to extract text from a PDF
@st.cache_data
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)  # Load page by page
            text += page.get_text()  # Extract text from each page
    return text


# Function to analyze the PDF and answer a question
@st.cache_resource
def analyze_pdf(pdf_path, question):
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Split the document into chunks
    doc = Document(page_content=pdf_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=3000)
    all_splits = text_splitter.split_documents([doc])
    uuids = [str(uuid4()) for _ in range(len(all_splits))]
    vector_store.add_documents(documents=all_splits, ids=uuids)

    # Retrieve the most relevant chunks
    retrieved_docs = vector_store.similarity_search(question, k=5)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Prepare the input for BioGPT
    input_text = f"Context: {docs_content}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate the answer using BioGPT
    outputs = model.generate(**inputs, max_length=300, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.split("Answer:")[-1].strip()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set the title of the app
st.title("PDF Chatbot")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the PDF..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        pdf_path = "NCCN_breastcancer.pdf"
        response = analyze_pdf(pdf_path, prompt)
    except Exception as e:
        response = f"An error occurred: {e}"

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
