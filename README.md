# Freeme - Local Notes Analyst

Freeme is a personal knowledge assistant that allows you to chat with your own notes stored locally on your computer. It uses the power of local Large Language Models (LLMs) via Ollama and Retrieval-Augmented Generation (RAG) with ChromaDB to provide answers grounded in your own documents. The interface is built with Streamlit.

## Features

*   **Local First:** Your notes and the analysis stay on your machine.
*   **Chat Interface:** Ask questions about your notes in natural language using Streamlit.
*   **RAG Pipeline:** Uses Langchain and ChromaDB to retrieve relevant information from your notes before generating an answer.
*   **Local LLMs:** Leverages Ollama to run LLMs like Llama 3 directly on your hardware.
*   **Multi-Format Support:** Supports various document types (`.txt`, `.md`, `.docx`, `.tex`, etc.) via the `unstructured` library (requires relevant dependencies).

## Setup and Installation

Follow these steps to get Freeme running:

1.  **Clone the Repository:**
    ```
    git clone <your-repository-url>
    cd freeme # Or your project folder name
    ```

2.  **Install System Dependencies:**
    *   **Ollama:** Download and install Ollama from [https://ollama.com/](https://ollama.com/).
    *   **Pandoc (for .tex support):** Install Pandoc. On macOS with Homebrew:
        ```
        brew install pandoc
        ```
    *   **libmagic (for file type detection):** Install libmagic. On macOS with Homebrew:
        ```
        brew install libmagic
        ```

3.  **Pull Ollama Models:**
    Make sure Ollama is running. Download the required LLM and embedding models (check `app.py` for the exact models configured, e.g., `LLM_MODEL` and `EMBEDDING_MODEL`):
    ```
    # Example models used in the current app.py
    ollama pull llama3:8b
    ollama pull nomic-embed-text
    ```

4.  **Create Python Virtual Environment:**
    ```
    python3 -m venv venv
    source venv/bin/activate
    # On Windows use: venv\Scripts\activate
    ```

5.  **Install Python Packages:**
    Install all required libraries from `requirements.txt`:
    ```
    pip install -r requirements.txt
    # You might need specific 'unstructured' extras depending on your file types:
    # Example: pip install "unstructured[docx,pdf]"
    ```

6.  **Create Notes Folder:**
    Create the folder where you will store your notes. **This folder (`documents/`) is ignored by Git.**
    ```
    mkdir documents
    ```
    **--> Place your personal notes (`.txt`, `.md`, `.docx`, `.tex`, etc.) inside this `documents` folder. <--**

## Running the Application

1.  Ensure Ollama is running in the background.
2.  Activate your virtual environment (`source venv/bin/activate`).
3.  Run the Streamlit app from your project's root directory:
    ```
    streamlit run app.py
    ```
    This will open the Freeme application in your web browser.

## Usage

1.  **Indexing:** The first time you run the app (or after clicking "Re-index Notes"), it will process the files in the `documents` folder and create a local vector database (`chroma_db_notes/`). This may take some time depending on the number and size of your notes.
2.  **Chat:** Once indexing is complete, you can ask questions about your notes in the chat input box.
3.  **Sources:** Expand the "Show Sources" section below the AI's answer to see which parts of your notes were used to generate the response.
4.  **Re-indexing:** If you add, modify, or delete notes in the `documents` folder, use the "Re-index Notes" button in the sidebar to update the application's knowledge base.

## Configuration

You can adjust key settings directly within `app.py`:

*   `LLM_MODEL`: The Ollama model tag for the main language model (e.g., `"llama3:8b"`).
*   `EMBEDDING_MODEL`: The Ollama model tag for generating embeddings (e.g., `"nomic-embed-text"`).
*   `NOTES_FOLDER`: The path to your notes folder (default: `"documents"`).
*   `PERSIST_DIRECTORY`: Where the ChromaDB index is stored (default: `"./chroma_db_notes"`).
*   Chunking parameters (`chunk_size`, `chunk_overlap`) in `split_documents`.
*   Number of retrieved documents (`k`) in `create_conversational_chain`.

## Core Dependencies

*   [Streamlit](https://streamlit.io/)
*   [Langchain](https://www.langchain.com/)
*   [Ollama](https://ollama.com/)
*   [ChromaDB](https://www.trychroma.com/)
*   [Unstructured](https://unstructured.io/)

---

