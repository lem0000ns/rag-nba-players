from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import openai
import os
import shutil
from scrape_hp import Scraper
import time
from queue import Queue
from threading import Thread

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(SCRIPT_DIR, "chroma")
DATA_PATH = os.path.join(SCRIPT_DIR, "hp_data")

def chunk_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

# clear hp_data after each batch
def clear_data_folder():
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
    os.makedirs(DATA_PATH)
    print(f"Cleared {DATA_PATH} folder")

def store_chroma_callback(vectorstore: Chroma):
    chunks = chunk_documents()
    if chunks:
        vectorstore.add_documents(chunks)
        print(f"Stored {len(chunks)} chunks in {CHROMA_PATH}")
    else:
        print("Warning: No chunks generated from documents")

    clear_data_folder()

def main():
    # clear temporary text file storage
    clear_data_folder()

    # clear chroma database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    # clear scraped_urls.json
    if os.path.exists("scraped_urls.json"):
        os.remove("scraped_urls.json")
        print("Cleared scraped_urls.json")
    # create new database from documents
    vectorstore = Chroma(collection_name="harry_potter_collection", persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

    before_time = time.time()
    
    scraper = Scraper(batch_size=20, store_callback=lambda: store_chroma_callback(vectorstore))

    def retrieve(q):
        while True:
            method_name = q.get()
            if method_name is None:
                break
            method = getattr(scraper, method_name)
            method()
            q.task_done()
    
    q = Queue(maxsize=0)
    num_workers = 8
    for i in range(num_workers):
        worker = Thread(target=retrieve, args=(q,))
        worker.daemon = True
        worker.start()
    
    for method_name in ['retrieve_places', 'retrieve_characters', 'retrieve_creatures', 'retrieve_novels', 'retrieve_magic', 'retrieve_things_one', 'retrieve_things_two', 'retrieve_events']:
        q.put(method_name)
    
    q.join()

    after_time = time.time()
    print(f"Time taken: {after_time - before_time} seconds")
    print(f"Scraped {scraper.documents_scraped} documents")

    clear_data_folder()

def print_num_documents_in_chroma():
    """Prints the number of documents stored in the Chroma vectorstore."""
    vectorstore = Chroma(collection_name="harry_potter_collection", persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    num_docs = vectorstore._collection.count()
    print(f"Number of documents in Chroma: {num_docs}")

if __name__ == "__main__":
    main()
    print_num_documents_in_chroma()