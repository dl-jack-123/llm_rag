# LLM + RAG Example

This is a simple example of implementing a Retrieval Augmented Generation (RAG) system using LangChain, FAISS, and Google's Gemini model.

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

The example code in `rag_example.py` demonstrates:
- Loading and splitting documents
- Creating embeddings using Vertex AI
- Storing vectors in FAISS
- Querying the RAG system using Gemini

To run the example:
```bash
python rag_example.py
```

以上範例拿 uv package 來當作例子> 通常 uv 會稱呼為紫外線, 但經過 rag 後, 回答為趨近預期中的答案