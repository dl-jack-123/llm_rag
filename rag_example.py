import os
from typing import List
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai

# Load environment variables
load_dotenv()

# Configure Google Generative AI
# genai.Client().models

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None
        self.qa_chain = None

    def load_documents(self, file_paths: List[str]):
        """Load documents from text files."""
        documents = []
        for path in file_paths:
            loader = TextLoader(path)
            documents.extend(loader.load())
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        
        # Initialize QA chain with Gemini
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-pro-exp",
            temperature=0,
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )

    def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        if not self.qa_chain:
            raise ValueError("Please load documents first using load_documents()")
        
        response = self.qa_chain.run(question)
        return response

def main():
    # Initialize RAG system
    rag = RAGSystem()
    
    # Example usage
    # First, create a sample document
    with open("sample.txt", "w") as f:
        f.write("""
        Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, 
        especially computer systems. These processes include learning (the acquisition of information 
        and rules for using the information), reasoning (using rules to reach approximate or definite 
        conclusions) and self-correction.

        Machine Learning (ML) is a subset of AI that provides systems the ability to automatically 
        learn and improve from experience without being explicitly programmed.
        """)
    
    # Load the document
    rag.load_documents(["sample.txt"])
    
    # Example query
    question = "What is the relationship between AI and ML?"
    print(f"Question: {question}")
    answer = rag.query(question)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
