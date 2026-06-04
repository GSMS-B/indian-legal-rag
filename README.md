---
title: Indian Legal RAG
emoji: 🏛️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
---

<h1 align="center">🏛️ Indian Criminal Law AI Assistant (BNS/BNSS/BSA)</h1>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/ChromaDB-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white" alt="ChromaDB">
  <img src="https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white" alt="Groq">
  <img src="https://img.shields.io/badge/OpenRouter-000000?style=for-the-badge&logo=openai&logoColor=white" alt="OpenRouter">
</div>

<br>

An advanced **Retrieval-Augmented Generation (RAG)** application designed to navigate, query, and explain the newly reformed Indian Criminal Justice system. It intelligently retrieves context across all three new legal codes to provide accurate, legally grounded answers.

### 📚 The Three Pillars of the Database
1. **BNS (Bharatiya Nyaya Sanhita, 2023):** Replaces the Indian Penal Code (IPC). Defines criminal offences and punishments.
2. **BNSS (Bharatiya Nagarik Suraksha Sanhita, 2023):** Replaces the Criminal Procedure Code (CrPC). Governs criminal procedures and police powers.
3. **BSA (Bharatiya Sakshya Adhiniyam, 2023):** Replaces the Indian Evidence Act (IEA). Dictates the rules of evidence in courts.

---

## ✨ Key Features

- 🔍 **Intelligent Semantic Search:** Uses `BAAI/bge-base-en-v1.5` embeddings to understand legal concepts, not just exact keywords.
- ⚖️ **Cross-Act Retrieval:** Automatically identifies if a query requires procedural context (BNSS), penal context (BNS), or evidentiary context (BSA), pulling minimum thresholds from each relevant act.
- ⚡ **Lightning Fast Generation:** Powered by `Llama-3` via Groq API for near-instant responses.
- 🛡️ **Robust Fallbacks:** Built-in API fallback chain (Groq 70B → Groq 8B → OpenRouter Auto) ensuring 100% uptime.
- 🖥️ **Interactive UI:** A beautiful, responsive Streamlit interface featuring one-click example queries, expanding source citations, and clear formatting.

---

## 🛠️ Architecture & Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Embeddings** | HuggingFace (`BAAI/bge-base-en-v1.5` via `sentence-transformers`) |
| **Vector Database** | ChromaDB (Local Persistent) |
| **LLM Engine** | Groq & OpenRouter (`llama-3.3-70b-versatile`) |

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the repository
```bash
git clone https://github.com/GSMS-B/indian-legal-rag.git
cd indian-legal-rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional but Recommended:
HF_TOKEN=your_huggingface_read_token_here
```
*(Note: `HF_TOKEN` is highly recommended. While the embedding models are free, providing a Hugging Face token prevents unauthenticated rate-limiting when the app tries to download the sentence-transformer weights).*

### 4. Build the Vector Database (One-time setup)
Because the vector database (`chroma_db/`) is too large for GitHub, you must build it locally the first time you run the project:
```bash
python scripts/01_chunk.py
python scripts/02_embed_store.py
```

### 5. Run the Application
```bash
streamlit run app.py
```

---

## 📝 License
This project is open source and available under the [MIT License](LICENSE).
