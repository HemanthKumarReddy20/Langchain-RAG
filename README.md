# 📚 RAG Application — Retrieval-Augmented Generation with LangChain & HuggingFace

A Retrieval-Augmented Generation (RAG) pipeline built with **LangChain**, **HuggingFace**, and **ChromaDB**. The application loads a PDF document, chunks and embeds it into a persistent vector store, and answers natural language questions using a LLaMA-based LLM with retrieved context.

---

## 🗂️ Project Structure

```
RAG_Application/
├── Data/
│   └── Personal_Biodata_Karthik.pdf   # Source document for RAG
├── Model/
│   ├── RAGModel.ipynb                 # Main Jupyter Notebook (RAG pipeline)
│   └── db/                            # Persisted ChromaDB vector store
├── .env                               # API keys (not committed)
├── .gitignore
└── requirements.txt
```

---

## ⚙️ How It Works

The pipeline follows the standard RAG architecture:

```
PDF Document
     │
     ▼
PDF Loader (PyPDFLoader)
     │
     ▼
Text Splitter (RecursiveCharacterTextSplitter)
  chunk_size=500, chunk_overlap=50
     │
     ▼
Embeddings (all-MiniLM-L6-v2)
     │
     ▼
ChromaDB Vector Store (persisted to disk)
     │
     ▼
MultiQueryRetriever (top-k=3)
     │
     ▼
Prompt Template + LLaMA-3.1-8B-Instruct
     │
     ▼
StrOutputParser → Final Answer
```

---

## 🤖 Models Used

| Component | Model |
|---|---|
| **LLM (RAG chain)** | `meta-llama/Llama-3.1-8B-Instruct` (temp=0.3) |
| **Chat model** | `meta-llama/Llama-3.1-8B-Instruct` (temp=0.7) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | ChromaDB (local persistence) |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd RAG_Application
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

> 🔑 Get your free API token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).  
> ⚠️ Make sure you have access to `meta-llama/Llama-3.1-8B-Instruct` (requires accepting the model license on HuggingFace).

### 5. Add Your Document

Place your PDF in the `Data/` directory and update the file path in the notebook:

```python
documents = document_loaders.PyPDFLoader(file_path='../Data/YourDocument.pdf')
```

### 6. Run the Notebook

```bash
jupyter notebook Model/RAGModel.ipynb
```

Run cells sequentially. The vector store is persisted to `Model/db/`, so document ingestion only needs to happen once.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain`, `langchain-core`, `langchain-community` | Core RAG framework |
| `langchain-huggingface` | HuggingFace LLM & embeddings integration |
| `langchain-chroma` | ChromaDB vector store integration |
| `huggingface-hub` | Model hub access |
| `transformers`, `sentence-transformers` | Model inference & embeddings |
| `chromadb` | Local vector database |
| `pypdf` | PDF loading |
| `python-dotenv` | Environment variable management |
| `pydantic` | Data validation |
| `ipykernel`, `ipywidgets` | Jupyter Notebook support |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 💬 Example Query

```python
response = chain.invoke('Who is Karthik? Explain what you know about Karthik.')
print(response)
```

The model retrieves the most relevant chunks from the vector store and generates a structured, professional response grounded in the document content.

---

## 🔍 Key Configuration

| Parameter | Value | Description |
|---|---|---|
| `chunk_size` | 500 | Characters per text chunk |
| `chunk_overlap` | 50 | Overlap between chunks to preserve context |
| `search_kwargs k` | 3 | Number of chunks retrieved per query |
| `max_new_tokens` | 512 | Maximum tokens in LLM response |
| `temperature (RAG)` | 0.3 | Lower = more factual responses |
| `temperature (Chat)` | 0.7 | Higher = more conversational responses |

---

## 📝 Notes

- The ChromaDB vector store is persisted in `Model/db/`. If you change documents, delete this folder and re-run the ingestion cells.
- The `MultiQueryRetriever` generates multiple variants of your query internally to improve retrieval coverage.
- The system prompt instructs the LLM to respond professionally and to clearly state when information is unavailable.

---

## 📄 License

This project is for educational and personal use.
