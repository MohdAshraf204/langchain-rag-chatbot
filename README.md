# LangChain RAG Chatbot

A small command-line chatbot that answers questions using Markdown and plain-text files. LangChain handles splitting, embeddings, retrieval, and chat prompts; Chroma stores vectors on your computer; the Google Gemini Developer API provides chat and embedding models.

This project defaults to `gemini-3.5-flash-lite` for chat and `gemini-embedding-2` for embeddings. 

Try these questions against the sample file:

- `What is RAG, and what are its main steps?`
- `Where is the vector database saved?`
- `What file types can this starter read?`
- `What is the capital of France?` (The chatbot should say the documents do not provide that information.)

Type `/exit` to end the chat. Conversation history is kept for the current run only; start a new process for a fresh conversation.
## About DrillBooks

[DrillBooks](https://www.drillbooks.codes/) is my website, where I sell hands-on practice resources for data professionals and people building data-analysis skills. I created this chatbot as a FAQ assistant for DrillBooks products. It uses the product information in `data/drillbooks_website_knowledge.md` to answer questions about the resources, their features, curriculum, requirements, purchase, and support. The RAG approach helps ground answers in the provided website information; when the documents do not contain an answer, the chatbot should say so instead of guessing.

