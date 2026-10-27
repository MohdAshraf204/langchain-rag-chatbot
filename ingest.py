import argparse
import shutil
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DATA_DIR, INDEX_MARKER, PROJECT_DIR, VECTOR_DB_DIR, make_embeddings, make_vector_store, require_api_key

def load_local_documents() -> list[Document]:
    """Read .md and .txt files while keeping each relative filename as metadata."""
    documents: list[Document] = []

    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": path.relative_to(DATA_DIR).as_posix()},
                )
            )

    return documents

def clear_existing_index() -> None:
    """Remove only the known project-local vector database before a rebuild."""
    expected_path = (PROJECT_DIR / "chroma_db").resolve()
    actual_path = VECTOR_DB_DIR.resolve()
    if actual_path != expected_path:
        raise SystemExit(f"Refusing to remove unexpected path: {actual_path}")
    if VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)

def main() -> None:
    parser = argparse.ArgumentParser(description="Index .md and .txt files from data/.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="replace the existing local Chroma index before indexing documents",
    )
    args = parser.parse_args()

    require_api_key()

    if INDEX_MARKER.exists() and not args.rebuild:
        raise SystemExit(
            "An index already exists. Use 'python ingest.py --rebuild' to replace it."
        )

    if args.rebuild:
        clear_existing_index()
    elif VECTOR_DB_DIR.exists() and any(VECTOR_DB_DIR.iterdir()):
        raise SystemExit(
            "A partial index already exists. Use 'python ingest.py --rebuild' to replace it."
        )

    documents = load_local_documents()
    if not documents:
        raise SystemExit(
            f"No non-empty .md or .txt files found in {DATA_DIR}. Add a document and try again."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise SystemExit("No text chunks were created. Check that your documents contain text.")

    vector_store = make_vector_store(make_embeddings())
    vector_store.add_documents(chunks)
    INDEX_MARKER.write_text("Index created successfully.\n", encoding="utf-8")

    print(f"Indexed {len(documents)} document(s) into {len(chunks)} chunks.")
    print(f"Local vector database: {VECTOR_DB_DIR}")
    print("You can now start the chatbot with: python chat.py")

if __name__ == "__main__":
    main()
