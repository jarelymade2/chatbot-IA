
import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_PATH = "./docs"
CHROMA_PATH = "./chroma_db"

def build_vectorstore():
    print(" Cargando documentos del catálogo Aracari Travel...")
    loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"    {len(documents)} documento(s) cargados")

    print("✂️  Dividiendo en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"    {len(chunks)} fragmentos creados")

    print(" Generando embeddings (primera vez tarda ~2 min)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print(" Guardando en ChromaDB...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"\n ¡Listo! {len(chunks)} fragmentos indexados en '{CHROMA_PATH}'")

if __name__ == "__main__":
    build_vectorstore()