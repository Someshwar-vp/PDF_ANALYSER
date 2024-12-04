# Install required libraries

import fitz  # PyMuPDF for PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Step 1: Extract text from the PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Specify the path to your breast cancer-related PDF
pdf_path = "NCCN_breastcancer.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

# Step 2: Split the extracted text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_text(pdf_text)

# Step 3: Use ClinicalBERT for embedding
modelPath = "emilyalsentzer/Bio_ClinicalBERT"  # ClinicalBERT model path

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

# Create FAISS database
db = FAISS.from_texts(docs, embeddings)

# Step 4: Use ClinicalBERT for QA
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

question_answerer = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    return_tensors="pt"
)

llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

retriever = db.as_retriever(search_kwargs={"k": 4})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever,
    return_source_documents=False
)

# Step 5: Ask questions to the model
question = "What are the symptoms of breast cancer?"  # Example question
answer = qa.run(question)
print("Answer:", answer)
