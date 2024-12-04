from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel
from langchain_huggingface import HuggingFacePipeline
import fitz  # PyMuPDF for document loading
import torch
from transformers import BioGptTokenizer, BioGptForCausalLM

# ------------------------------
# Document Loading with PyMuPDF
# ------------------------------

def load_pdf(file_path):
    """Load text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Example PDF path
pdf_path = "NCCN_breastcancer.pdf"
document_content = load_pdf(pdf_path)

# Split the content into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.create_documents([document_content])

# ---------------------------------
# Text Embedding with Bio_ClinicalBERT
# ---------------------------------

from langchain.embeddings.base import Embeddings
from typing import List

class CustomClinicalBERTEmbeddings(Embeddings):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", device="cpu", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.max_length = max_length

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text).tolist() for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text).tolist()

    def _get_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze()

# Initialize embeddings
embeddings = CustomClinicalBERTEmbeddings()

# Create FAISS vector store
db = FAISS.from_documents(docs, embeddings)

# -------------------------
# LLM Model: Using BioGPT
# -------------------------

# Load BioGPT model and tokenizer
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

# Create text-generation pipeline
question_answerer = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,  # Room for output generation
    truncation=True,
    do_sample=False
)

# Use the updated HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=question_answerer)

# Test the pipeline
test_response = question_answerer("What is breast cancer?")
print("Pipeline Output:", test_response)

# Create RetrievalQA chain
retriever = db.as_retriever(search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever,
    return_source_documents=False
)

# Test the QA chain
question = "What is breast cancer?"
try:
    result = qa.invoke({"query": question})  # Updated to `invoke`
    print("Answer:", result)
except Exception as e:
    print("Error during QA:", e)



