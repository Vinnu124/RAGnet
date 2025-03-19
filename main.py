import asyncio
import random
import sys
from pathlib import Path
from base.chain import ask_question, create_chain
from base.config import Config
from base.ingestor import Ingestor
from base.model import create_llm
from base.retriever import create_retriever

LOADING_MESSAGES = [
    "Hmmm....let me think about that.",
    "I'm searching for the answer...",
    "Give me a moment to find the information...",
    "I'm on it!",
    "Let me find that for you...",
]

class TerminalRAG:
    def __init__(self):
        self.config = Config()
        self.messages = []
        self.chain = None
        self.vector_store = None
        self.llm = None
        self.retriever = None

    def initialize_system(self, files):
        """Initialize the RAG system with PDF files"""
        print("\n🔍 Initializing RAG system...")
        file_paths = [Path(f) for f in files]
        
        if not file_paths:
            raise ValueError("No PDF files provided")
            
        self.vector_store = Ingestor().ingest(file_paths)
        print('ingested')
        self.llm = create_llm()
        print('llm created')
        self.retriever = create_retriever(self.llm, vector_store=self.vector_store)
        print('retriever created')
        self.chain = create_chain(self.llm, self.retriever)
        print('chain created')
        print("✅ System ready!")

    async def process_question(self, question: str):
        """Process a question and return the answer with sources"""
        full_response = ""
        documents = []
        
        print(f"\n💡 {random.choice(LOADING_MESSAGES)}")
        
        async for event in ask_question(self.chain, question, session_id="terminal"):
            if isinstance(event, str):
                full_response += event
                print(event, end="", flush=True)
            elif isinstance(event, list):
                documents.extend(event)
                
        print("\n\n📚 Sources:")
        for i, doc in enumerate(documents, 1):
            print(f"\n🔗 Source #{i}")
            print("-" * 50)
            print(doc.page_content)
            print("-" * 50)
            
        return full_response

    def chat_loop(self):
        """Terminal-based chat interface"""
        print("\n" + "="*50)
        print("🤖 RagBase Terminal Interface")
        print("="*50)
        print("Type 'exit' to quit\n")
        
        while True:
            try:
                question = input("\n💬 Your question: ")
                if question.lower() in ['exit', 'quit']:
                    break
                    
                response = asyncio.run(self.process_question(question))
                self.messages.append({"role": "user", "content": question})
                self.messages.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                print("\n🚫 Session interrupted")
                break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python terminal_rag.py [path/to/pdf1] [path/to/pdf2] ...")
        sys.exit(1)
        
    rag = TerminalRAG()
    try:
        rag.initialize_system(sys.argv[1:])
        rag.chat_loop()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
