# RAG chatbot starter notes

This starter project uses retrieval-augmented generation (RAG) to answer questions from local Markdown and plain-text documents.

## What happens during indexing?

The ingestion script finds `.md` and `.txt` files inside the `data` folder and its subfolders. It splits the text into chunks of about 800 characters, with 120 characters of overlap between neighboring chunks. An embedding model converts each chunk into a vector, and Chroma saves the vectors in the local `chroma_db` folder.

## What happens when someone asks a question?

The chatbot converts the question into an embedding, searches Chroma for the four most similar chunks, and includes those passages in a prompt to the chat model. The model is asked to answer from the retrieved passages and to say when the documents do not contain the answer. The chatbot also prints the source filenames so the reader can inspect the evidence.

## Useful project facts

- The vector database is stored locally in `chroma_db`.
- The command-line chat history is kept only while that chat process is running.
- This starter supports `.md` and `.txt` files. It does not parse PDF files yet.
- Adding or changing a document requires rebuilding the index with `python ingest.py --rebuild`.
- RAG can make answers better grounded, but it cannot guarantee that every response is correct.

