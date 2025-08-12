import streamlit as st
import os
import json
import pymupdf  # fitz
import time
import pytesseract
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA, StuffDocumentsChain
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pdf2image import convert_from_path
from langchain.docstore.document import Document

# ===================================
# Funciones auxiliares
# ===================================
def load_pdf(file):
    # Guardar temporalmente el PDF
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())

    # Intentar extracción normal
    loader = PyPDFium2Loader("temp.pdf")
    docs = loader.load()

    filtered_docs = [doc for doc in docs if doc.page_content.strip() != "" and "about:blank" not in doc.page_content.lower()]
    
    if filtered_docs:
        st.write(f"Páginas con texto útil: {len(filtered_docs)}")
        return filtered_docs
    
    # Si no hay texto, pasar a OCR
    st.warning("No se encontró texto en el PDF. Usando OCR, esto puede tardar más...")
    pages = convert_from_path("temp.pdf")
    ocr_docs = []

    for i, page_img in enumerate(pages):
        text = pytesseract.image_to_string(page_img, lang="spa")  # cambiar lang si es inglés u otro idioma
        if text.strip():
            ocr_docs.append(Document(page_content=text, metadata={"page": i+1, "source": "OCR"}))

    if not ocr_docs:
        st.error("OCR completado, pero no se pudo extraer texto legible.")
        return []

    st.success(f"OCR completado: {len(ocr_docs)} páginas con texto.")
    return ocr_docs

def get_pdf_first_page_image(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    doc = pymupdf.open("temp.pdf")
    os.makedirs("static", exist_ok=True)
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
    return vector.as_retriever(search_type="similarity", search_kwargs={"k": 15})

def build_prompt():
    prompt = """
    Act as a software engineering researcher and help me conduct a systematic literature review.
    Read the following article and answer the four yes-or-no questions below. For each question, provide your response using the following format:

    Question: [The question]
    Answer format: Yes/No – [Exact excerpt from the article that supports the answer] – [Brief reasoning explaining why the excerpt supports the answer]

    Please ensure that the excerpt is quoted directly from the article and the reasoning clearly explains how the excerpt justifies your binary answer.

    Questions:
            1. Is the given article within the defined publication timeframe? (2015–2025) (Yes/No)
            2. Is the study empirical (e.g., experiment, survey, case study)? (Yes/No)
            3. Is the study theoretical? (Yes/No)
            4. Is the publication peer-reviewed or published in a reputable source? (Yes/No)

    Context: {context}"""
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

def get_llm(provider, api_key=None, model=None):
    if provider == "Ollama (Local)":
        llm = Ollama(model=model or "deepseek-r1:1.5b")
        st.write(f"💻 LLM local '{model}' cargado con Ollama.")
    
    elif provider == "OpenAI (Online)":
        if not api_key:
            st.error("❌ Falta la API key de OpenAI.")
            st.stop()
        llm = ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=0,
            openai_api_key=api_key
        )
        st.write(f"☁ LLM online '{model}' cargado con OpenAI.")
    
    elif provider == "Anthropic (Online)":
        if not api_key:
            st.error("❌ Falta la API key de Anthropic.")
            st.stop()
        llm = ChatAnthropic(
            model=model or "claude-3-5-sonnet-20240620",
            temperature=0,
            anthropic_api_key=api_key
        )
        st.write(f"☁ LLM online '{model}' cargado con Anthropic.")
    
    else:
        st.error("Proveedor no soportado.")
        st.stop()
    
    return llm

# ===================================
# App principal
# ===================================
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("🚀 PDF Analyzer con LLM Local/Online")

    # Configuración del LLM en la barra lateral
    with st.sidebar:
        st.subheader("⚙ Configuración del LLM")
        provider = st.selectbox(
            "Proveedor de LLM",
            ["Ollama (Local)", "OpenAI (Online)", "Anthropic (Online)"]
        )

        api_key = None
        model = None
        
        if provider != "Ollama (Local)":
            api_key = st.text_input("API Key", type="password")
        
        if provider == "Ollama (Local)":
            model = st.text_input("Modelo local", value="deepseek-r1:1.5b")
        elif provider == "OpenAI (Online)":
            model = st.text_input("Modelo OpenAI", value="gpt-4o-mini")
        elif provider == "Anthropic (Online)":
            model = st.text_input("Modelo Anthropic", value="claude-3-5-sonnet-20240620")

        uploaded_file = st.file_uploader("📄 Sube un PDF", type="pdf")

        if uploaded_file:
            try:
                image_path = get_pdf_first_page_image(uploaded_file)
                st.image(image_path, caption="Vista previa de la primera página", use_container_width=True)
            except Exception as e:
                st.error("Error al cargar la vista previa: " + str(e))

    # Procesar PDF y ejecutar QA
    if uploaded_file:
        with st.spinner("🔄 Procesando PDF..."):
            retriever = process_pdf(uploaded_file)
        
        llm = get_llm(provider, api_key, model)
        qa_chain = build_qa_chain(retriever, llm)
        
        if st.button("Generar respuesta"):
            with st.spinner("🤖 Generating response..."):
                start_time = time.time()
                response = qa_chain.invoke({"query": "Analyze the complete PDF and answer the given questions."})["result"]

                end_time = time.time()
                elapsed_time = end_time - start_time
                
                st.write("### 📜 Answer:")
                st.write(response)
                st.write(f"⏱️ Response generated in {elapsed_time:.2f} seconds")

    else:
        st.info("📥 Por favor, sube un archivo PDF para comenzar.")
