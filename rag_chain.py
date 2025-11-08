from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import openai
import os
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from evaluate import get_llm

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "./chroma"

RAG_PROMPT_INF = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

RAG_PROMPT_EVAL = (
    "You are a helpful assistant that answers multiple-choice questions about Harry Potter related lore. "
    "Use the following pieces of retrieved context to answer the multiple-choice question. "
    "Only output the letter of the correct answer (\"A\", \"B\", \"C\", or \"D\")."
    "DO NOT output any other text or characters."
    "\n\n"
    "Context: {context}"
)

def print_docs_information(query_results):
    print("\nLength of query results: ", len(query_results))
    for i, result in enumerate(query_results):
        print(f"Relevance score of result {i+1}: ", result[1])
        print(f"Content of result {i+1}: ", result[0])
        print("-" * 100)

def get_retriever(mode="inf", use_raw_vllm=False):
    """Get retriever and optionally LLM chain components.
    
    Args:
        mode: "inf" for inference or "eval" for evaluation (affects prompt)
        return_chain: If True, returns (retriever, chain). If False, returns just retriever.
        use_raw_vllm: If True and return_chain is True, returns (retriever, raw_vllm_instance) 
                      instead of (retriever, langchain_chain).
    
    Returns:
        - If return_chain=False: ensemble_retriever
        - If return_chain=True and use_raw_vllm=False: (ensemble_retriever, question_answer_chain)
        - If return_chain=True and use_raw_vllm=True: (ensemble_retriever, raw_vllm_instance)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
    vectorstore = Chroma(
        collection_name="harry_potter_collection", 
        persist_directory=CHROMA_PATH, 
        embedding_function=OpenAIEmbeddings()
    )
    
    # dense retrieval with vector embeddings
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # sparse retrieval using BM25
    raw = vectorstore.get()
    documents = [Document(page_content=text, metadata=meta) for text, meta in zip(raw['documents'], raw['metadatas'])]
    keyword_retriever = BM25Retriever.from_documents(documents)
    keyword_retriever.k = 5
    
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever, keyword_retriever], weights=[0.5, 0.5])
    
    # return raw vLLM if requested
    if use_raw_vllm:
        llm_instance = get_llm()
        return ensemble_retriever, llm_instance
    
    # otherwise, build LangChain chain
    if mode == "inf":
        rag_prompt = RAG_PROMPT_INF
    elif mode == "eval":
        rag_prompt = RAG_PROMPT_EVAL
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rag_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm=ChatOpenAI(model_name='gpt-4o-mini'), prompt=rag_prompt)
    
    return ensemble_retriever, question_answer_chain

def get_qa_chain(mode="inf"):
    ensemble_retriever, question_answer_chain = get_retriever(mode)
    # create retrieval chain
    rag_chain = create_retrieval_chain(ensemble_retriever, question_answer_chain)
    return rag_chain

def main():
    rag_chain = get_qa_chain()
    
    chat_history = [
        HumanMessage(content="Who is Lebron James?"),
        AIMessage(content="Lebron James is a basketball player.")
    ]
    
    response2 = rag_chain.invoke({
        "input": "What high school did he go to?",
        "chat_history": chat_history
    })
    print(response2["answer"])

if __name__ == "__main__":
    main()
