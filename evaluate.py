from datasets import load_dataset
from dotenv import load_dotenv
import vllm
import os
from vllm import SamplingParams
import torch
import gc

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
llm = None
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def generate_text(chat_input, llm_instance, max_tokens=1):
    """Generate text from chat input using the LLM.
    
    Args:
        chat_input: Chat messages in format [{"role": "system/user", "content": "..."}]
        llm_instance: The LLM instance to use for generation
    
    Returns:
        Generated text string
    """
    SAMPLING_PARAMS = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=max_tokens)
    output = llm_instance.chat(chat_input, SAMPLING_PARAMS)
    return output[0].outputs[0].text

def get_llm():
    """Get or initialize the LLM instance."""
    global llm
    if llm is None:
        llm = vllm.LLM(model=MODEL_NAME, tensor_parallel_size=1, dtype='half', max_model_len=8192)
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

def run_evaluation():
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

        model_answer = generate_text(chat_input, llm)
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
    run_evaluation()