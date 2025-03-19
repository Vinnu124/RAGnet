# RAGnet

RAGnet is an AI-powered document question-answering system that uses Retrieval Augmented Generation (RAG) to provide accurate responses based on the content of uploaded PDF documents.

## Features

- PDF document ingestion and processing
- Semantic chunking for optimal text splitting
- Vector storage using Qdrant for efficient retrieval
- Hugging Face language model integration (google/gemma-2-27b quantized to fp4)
- Reranking and chain filtering options for improved accuracy
- Session-based chat history management

## Installation

1. Clone the repository:
     ```sh
     git clone https://github.com/Vinnu124/RAGnet
     cd RAGnet
     ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up environment variables:
   - Get the QDrant URL to your cluster using ```https://qdrant.to/cloud```
   - Create a `.env` file in the project root which will contain the QDrant Link.
   - Add `QDRANT_URL` variable pointing to the project directory.

## Usage

Run the main application:

```sh
python main.py path/to/pdf1.pdf path/to/pdf2.pdf
```

To exit the application, type ```exit``` or ```quit```.

## Configuration

Adjust settings in `config.py`:

- Model parameters (embeddings, reranker, language model)
- Retriever settings (use of reranker, chain filter)
- Database and file paths

## File Structure

- `main.py`: Entry point of the application
- `config.py`: Configuration settings
- `ingestor.py`: Document processing and vector store creation
- `retriever.py`: Document retrieval logic
- `model.py`: Language model and embedding configurations
- `chain.py`: Question-answering chain setup
- `session_history.py`: Chat history management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

