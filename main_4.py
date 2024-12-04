import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Step 1: Load PDF Content using PyMuPDF
def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text

pdf_path = "NCCN_breastcancer.pdf"
pdf_content = load_pdf(pdf_path)

# Step 2: Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.create_documents([pdf_content])

# Step 3: Create Embeddings and Vector Store
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)
db = FAISS.from_documents(docs, embeddings)

# Step 4: Load the Question-Answering Model
model_name = "Intel/dynamic_tinybert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
question_answerer = pipeline("question-answering", model=model_name, tokenizer=tokenizer)

# Step 5: Wrap QA Pipeline with LangChain
llm = HuggingFacePipeline(pipeline=question_answerer)

# Step 6: Create Retriever and QA Chain
retriever = db.as_retriever(search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)

# Step 7: Ask a Question
question = "What is the main topic of the PDF?"
result = qa.invoke({"query": question})
print(result)

