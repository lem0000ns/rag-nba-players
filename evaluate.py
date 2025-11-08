from datasets import load_dataset
from dotenv import load_dotenv
import vllm
import os
from vllm import SamplingParams
import torch
import gc
from langchain_community.llms import VLLM
import re
import argparse

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
llm = None

def generate_text(chat_input, llm_instance, max_tokens=2, temperature=0.0, top_p=0.95):
    """Generate text from chat input using the LLM.
    
    Args:
        chat_input: Chat messages in format [{"role": "system/user", "content": "..."}]
        llm_instance: The LLM instance to use for generation
    
    Returns:
        Generated text string
    """
    SAMPLING_PARAMS = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    output = llm_instance.chat(chat_input, SAMPLING_PARAMS)
    return output[0].outputs[0].text

def get_llm():
    """Get or initialize the LLM instance."""
    global llm
    if llm is None:
        llm = vllm.LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            max_model_len=8192,
            dtype="half",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            disable_log_stats=True,
        )
    return llm

def cleanup():
    """Clean up GPU memory by deleting the LLM instance."""
    global llm
    if llm is not None:
        print("Cleaning up GPU memory for LLM instance...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared")
            del llm
            llm = None
            print("LLM instance deleted")
            gc.collect()
        except Exception as e:
            print(f"Error during cleanup: {e}")

def extract_answer_letter(text):
    """Extract the answer letter (A, B, C, or D) from model output.
    
    Args:
        text: Model output text
        
    Returns:
        Single letter (A, B, C, or D) or original text if no match found
    """
    # Try to find a single letter answer
    text = str(text).strip().upper()
    
    # Look for patterns like "A)", "(A)", "A.", "A:", or just "A"
    match = re.search(r'\b([ABCD])\b', text)
    if match:
        return match.group(1)
    
    # If no match, return first character if it's A, B, C, or D
    if text and text[0] in ['A', 'B', 'C', 'D']:
        return text[0]
    
    return text

def run_evaluation_rag(temperature=0.0, top_p=0.95):
    """Evaluate the RAG pipeline on the Harry Potter quiz dataset using raw vLLM."""
    from rag_chain import get_retriever
    
    # Get retriever and raw vLLM instance for strict token control
    ensemble_retriever, llm_instance = get_retriever(mode="eval", use_raw_vllm=True)
    
    ds = load_dataset("cross-ling-know/HarryPotter-Quiz", split="English")
    correct = 0
    total = len(ds)

    for i, item in enumerate(ds):
        question = item['question']
        option_a = item['A']
        option_b = item['B']
        option_c = item['C']
        option_d = item['D']
        ground_truth = item['GT']

        # Format the question with options
        formatted_question = (
            f"{question}\n"
            f"A) {option_a}\n"
            f"B) {option_b}\n"
            f"C) {option_c}\n"
            f"D) {option_d}"
        )

        try:
            # Retrieve relevant documents
            retrieved_docs = ensemble_retriever.invoke(formatted_question)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Build the prompt with context
            chat_input = [
                {
                    "role": "system", 
                    "content": (
                        "You are a helpful assistant that answers multiple-choice questions about Harry Potter. "
                        "Use the following context to answer the question. "
                        "Only output the letter of the correct answer (A, B, C, or D). "
                        "Do NOT output any other text."
                    )
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion:\n{formatted_question}\n\nAnswer:"
                }
            ]
            
            # Generate with strict max_tokens=2
            model_answer = generate_text(chat_input, llm_instance, temperature=temperature, top_p=top_p)
            
            # Extract just the letter from the response
            extracted_answer = extract_answer_letter(model_answer)
            
            # Check if correct
            is_correct = extracted_answer.upper() == ground_truth.upper()
            if is_correct:
                correct += 1
            
            # Print progress every 10 questions
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{total} | Accuracy so far: {correct / (i + 1):.2%}")
            
            # Print incorrect answers for debugging
            if not is_correct:
                print(f"Question: {question}")
                print(f"Options: {option_a}, {option_b}, {option_c}, {option_d}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Model Answer: {model_answer}")
                print("-" * 100)
        
        except Exception as e:
            print(f"\n⚠️  Error on question {i + 1}: {e}")
            print(f"Question: {question}")
            print("-" * 100)

    accuracy = correct / total
    print(f"\n{'='*100}")
    print(f"FINAL RESULTS")
    print(f"{'='*100}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"{'='*100}\n")

def run_evaluation_vanilla(temperature=0.0, top_p=0.95):
    """Evaluate vanilla LLM (no RAG) on the Harry Potter quiz dataset."""
    llm = get_llm()
    ds = load_dataset("cross-ling-know/HarryPotter-Quiz", split="English")

    correct = 0
    total = len(ds)

    for item in ds:
        question = item['question']
        option_a = item['A']
        option_b = item['B']
        option_c = item['C']
        option_d = item['D']
        ground_truth = item['GT']

        chat_input = [
            {"role": "system", "content": "You are a helpful assistant that answers questions about Harry Potter related lore. You will be given a question and four options. Only output the letter of the correct option, with no additional text or formatting. Example: \"A\" or \"B\" or \"C\" or \"D\""},
            {"role": "user", "content": f"Question: {question}\nOptions: {option_a}, {option_b}, {option_c}, {option_d}\nAnswer:"}
        ]

        model_answer = generate_text(chat_input, llm, temperature=temperature, top_p=top_p)
        if str(model_answer).lower() == str(ground_truth).lower():
            correct += 1
        else:
            print(f"Question: {question}")
            print(f"Options: {option_a}, {option_b}, {option_c}, {option_d}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Model Answer: {model_answer}")
            print("-" * 100)

    print(f"Accuracy: {correct / total}")
    cleanup()
        
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--evaluation", type=str, default="rag", choices=["rag", "vanilla"])
    argparser.add_argument("--temperature", type=float, default=0.0)
    argparser.add_argument("--top_p", type=float, default=0.95)
    args = argparser.parse_args()
    if args.evaluation == "rag":
        run_evaluation_rag(args.temperature, args.top_p)
    elif args.evaluation == "vanilla":
        run_evaluation_vanilla(args.temperature, args.top_p)
    else:
        print("Invalid evaluation type")