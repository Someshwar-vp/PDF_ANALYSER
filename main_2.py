from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel 
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import fitz  # PyMuPDF for document loading
from sentence_transformers import SentenceTransformer, models, losses
import torch

# ------------------------------
# Document Loading with PyMuPDF
# ------------------------------

# Function to load text from a PDF file
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Example PDF path (update with your actual file path)
pdf_path = "NCCN_breastcancer.pdf"
document_content = load_pdf(pdf_path)

# Split the loaded content into chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.create_documents([document_content])

# ---------------------------------
# Text Embedding with Bio_ClinicalBERT
# ---------------------------------

# Define the path to the ClinicalBERT model
embedding_model_path = "emilyalsentzer/Bio_ClinicalBERT"

# Initialize embeddings using ClinicalBERT
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

# Create a FAISS vector store from the documents
db = FAISS.from_documents(docs, embeddings)

# -------------------------
# LLM Model: Using biogpt
# -------------------------

# Define the path to the biogpt model
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Define the pipeline for the LLM
question_answerer = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_tensors=True,
    max_length=512,
    temperature=0.7
)

# Wrap the pipeline in a LangChain-compatible class
llm = HuggingFacePipeline(pipeline=question_answerer)

# ---------------------------
# Retrieval and Question-Answering
# ---------------------------

# Create a retriever from the FAISS database
retriever = db.as_retriever(search_kwargs={"k": 4})

# Create a RetrievalQA chain using LLaMA 2
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever,
    return_source_documents=False
)

# -----------------
# Ask a Question
# -----------------

question = "what is breast cancer?"
result = qa.run({"query": question})
print("Answer:", result)
