import streamlit as st
import os
import json
import pymupdf  # fitz
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA, StuffDocumentsChain
from langchain_community.llms import Ollama

def load_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    loader = PyPDFLoader("temp.pdf")
    return loader.load()

def get_pdf_first_page_image(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    doc = pymupdf.open("temp.pdf")
    os.makedirs("static", exist_ok=True)  # Ensure storage directory exists
    pix = doc[0].get_pixmap()
    image_path = "static/first_page.png"
    pix.save(image_path)
    return image_path

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

def create_retriever(docs_json):
    docs = json.loads(docs_json)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.write("Embeddings loaded successfully.")
    vector = FAISS.from_texts(
        [doc["page_content"] for doc in docs],
        embedder,
        metadatas=[{"source": doc["metadata"].get("source", "Uploaded PDF")} for doc in docs]
    )
    st.write("FAISS vector store created successfully.")
    return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def build_prompt():
    prompt = """
    1. Act as a software engineering researcher and help me conduct a systematic literature review.
    1. You have to analyze the following context.
    2. To analyze that context I'm giving you some Yes or No questions.
    3. Give the answer with the next structure, answering each question without the reasoning:
            Paper Title
                Question 1: Yes or No
                Question 2: Yes or No
                Question 3: Yes or No
                Question 4: Yes or No
                Question 5: Yes or No
                Question 6: Yes or No
                Question 7: Yes or No
                Question 8: Yes or No
                Question 9: Yes or No
                Question 10: Yes or No
                Question 11: Yes or No
                Question 12: Yes or No
                Question 13: Yes or No
                Question 14: Yes or No
                Question 15: Yes or No

    Questions:
            1. Is the article relevant to the research question? (Yes/No)
            2. Is the article published in a peer-reviewed journal or conference? (Yes/No)
            3. Is the article within the defined publication timeframe? (2015–2025) (Yes/No)
            4. Is the article written in English? (Yes/No)
            5. Does the article provide a clear research methodology? (Yes/No)
            6. Is the study empirical (e.g., experiment, survey, case study) or theoretical? (Empirical/Theoretical)
            7. Are the data sources and sample sizes adequate? (Yes/No)
            8. Does the article address software engineering topics relevant to the study? (Yes/No)
            9. Does the article discuss challenges or solutions in software engineering? (Yes/No)
            10. Is there a clear contribution (e.g., new framework, model, or empirical evidence)? (Yes/No)
            11. Does the article include comparative analysis with existing work? (Yes/No)
            12. Is the article cited frequently (indicating impact)? (Yes/No)
            13. Is the study conducted by reputable authors or institutions? (Yes/No)
            14. Are the conclusions well-supported by evidence? (Yes/No)
            15. Should this article be included in the final review? (Yes/No)

    Context: {context}

    Helpful Answer:"""
    return PromptTemplate.from_template(prompt)

def build_qa_chain(retriever, llm):
    prompt = build_prompt()
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="Context:\ncontent:{page_content}\n",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
    )
    return RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        return_source_documents=True,
    )

def process_pdf(file):
    docs = load_pdf(file)
    chunks = chunk_documents(docs)
    retriever = create_retriever(json.dumps([{"page_content": d.page_content, "metadata": d.metadata} for d in chunks]))
    return retriever

def get_llm():
    llm = Ollama(model="deepseek-r1:1.5b")
    st.write("LLM loaded successfully.")
    return llm

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("🚀 Fast RAG-based QA with DeepSeek R1")
    
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        
        if uploaded_file:
            try:
                image_path = get_pdf_first_page_image(uploaded_file)
                st.image(image_path, caption="First Page Preview", use_column_width=True)
            except Exception as e:
                st.error("Failed to load preview: " + str(e))
    
    if uploaded_file:
        with st.spinner("🔄 Processing PDF..."):
            retriever = process_pdf(uploaded_file)
        
        llm = get_llm()
        qa_chain = build_qa_chain(retriever, llm)
        
        user_input = st.text_input("Press enter to go:")
        
        if user_input:
            with st.spinner("🤖 Generating response..."):
                response = qa_chain.invoke({"query": user_input})["result"]
                st.write("### 📜 Answer:")
                st.write(response)
    else:
        st.info("📥 Please upload a PDF file to proceed.")










