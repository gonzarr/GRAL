# 🚀 PDF Analyzer con LLM Local/Online

Analiza artículos académicos en PDF y responde cuatro preguntas binarias (Yes/No) justificadas con **citas textuales** extraídas del documento.

## 📦 Características
- Vista previa de la primera página del PDF.  
- Extracción de texto por `PyPDFium2Loader` y fallback por **OCR** (`pytesseract`) si no hay texto.  
- Chunking automático (`RecursiveCharacterTextSplitter`).  
- Embeddings con `sentence-transformers/all-MiniLM-L6-v2` y búsqueda semántica usando `FAISS`.  
- Soporta LLM local (**Ollama**) o LLMs online (**OpenAI**, **Anthropic**).  
- Prompt diseñado para responder 4 preguntas académicas con formato:  
  `Yes/No – "Cita exacta" – Razonamiento breve`.

## 🛠 Requisitos e instalación
Instala dependencias Python:
```bash
pip install streamlit pymupdf pytesseract pdf2image langchain langchain-community langchain-openai langchain-anthropic sentence-transformers faiss-cpu
```

Dependencias del sistema:
- **Tesseract OCR**
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- **Poppler** (requerido por `pdf2image`)
  - Ubuntu/Debian: `sudo apt-get install poppler-utils`
- **Ollama** (opcional, para LLM local): sigue instrucciones en https://ollama.ai

## ▶️ Cómo ejecutar
1. Lanza la app:
```bash
streamlit run app.py
```
2. En la barra lateral:
   - Elige proveedor del LLM: `Ollama (Local)`, `OpenAI (Online)` o `Anthropic (Online)`.
   - Si eliges un proveedor online, pega tu **API Key**.
   - Ajusta el nombre del modelo si lo deseas.
3. Sube un PDF (`.pdf`). Espera la vista previa de la primera página.
4. Presiona **Generar respuesta** para recibir las 4 respuestas con citas y justificación.

## 🔁 Flujo interno (resumen)
1. Guardado temporal del PDF (`temp.pdf`).  
2. Extracción con `PyPDFium2Loader`. Si no hay texto, OCR página a página con `pytesseract`.  
3. División en chunks (500 tokens aprox., overlap 50).  
4. Creación de embeddings (`HuggingFaceEmbeddings`) y FAISS vectorstore.  
5. Creación de un `retriever` y construcción de una cadena `RetrievalQA` con un prompt especializado.  
6. LLM responde y la app muestra la salida junto con el tiempo transcurrido.

## 🧭 Formato esperado de la respuesta
Cada pregunta debe devolver:
```
Question: [Pregunta]
Answer format: Yes/No – "Cita exacta del artículo" – [Razonamiento breve]
```
Ejemplo:
```
Question: Is the study empirical?
Answer format: Yes – "We performed a survey of 300 participants..." – El texto describe un muestreo y análisis empírico.
```

## ⚠️ Consejos y notas
- Cambia `pytesseract.image_to_string(..., lang="spa")` si el PDF está en otro idioma.  
- OCR en PDFs largos puede tardar — ten paciencia.  
- La calidad de la respuesta depende de la extracción de texto y del modelo LLM elegido.  
- Si usas OpenAI/Anthropic, asegúrate de que tu API Key tenga permisos y saldo suficiente.

