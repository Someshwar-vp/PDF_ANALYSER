# Install necessary libraries
# pip install -q langchain langchain-huggingface transformers pymupdf sentence-transformers faiss-cpu

# Import necessary modules
import fitz  # PyMuPDF
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from langchain.chains import RetrievalQA

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Path to your PDF file
pdf_path = "NCCN_breastcancer.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

# Step 2: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_text(pdf_text)

# Step 3: Initialize ClinicalBERT
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Step 4: Create embeddings with ClinicalBERT
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Step 5: Store document embeddings in FAISS
db = FAISS.from_texts(docs, embeddings)

# Step 6: Define the HuggingFace QA pipeline
# Define a wrapper to format the question-answering input
question_answerer = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=-1  # Use CPU
)

def huggingface_pipeline_wrapper(input_data):
    question = input_data.get("question", "")
    context = input_data.get("context", "")
    return question_answerer({"question": question, "context": context})

# Wrap the QA pipeline with LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=huggingface_pipeline_wrapper)

# Step 7: Initialize retriever
retriever = db.as_retriever(search_kwargs={"k": 4})

# Step 8: Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)

# Step 9: Ask a question
query = "What are the symptoms of breast cancer?"
results = retriever.get_relevant_documents(query)

# Combine retrieved documents to form a single context for the QA model
context = " ".join([doc.page_content for doc in results])

# Run the QA pipeline
response = llm({"question": query, "context": context})
print(f"Answer: {response}")

