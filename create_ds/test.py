from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import openai
from evaluate.eval import get_llm, generate_text
import json
import pandas as pd

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../chroma")

QA_GENERATION_PROMPT = """
You are an expert in the domain of Harry Potter, with extensive knowledge of the history, magic, characters, events, and lore of the Harry Potter universe. Your task is to extract or create 1 to 10 question-answer pairs from the provided source material. The title of the source material is also provided for context. The QAs will be used as a Harry Potter knowledge base for readers to test their understanding of the Harry Potter universe.

Each QA pair must include:

1. Question
- A clear and standalone question.
- The question must contain sufficient background information or context so that one can fully understand and attempt it without referring to the original source material.
- Define any abbreviations or domain-specific terminology used.
- DO NOT use phrases like "According to the source material" or "From the source material".
- The question difficulty should range from easy to hard. Easy questions should be surface-level, straightforward, and often answered with a single phrase or word. Hard questions should be more complex and require more reasoning.

2. Options
- Provide four answer choices.
- Only one option should be correct.
- The three incorrect options should be plausible but clearly wrong upon reviewing the source material and careful reasoning. If the question requires reasoning, these incorrect options should ideally be derived by subtlely altering the logic or assumptions behind the correct answer. Otherwise, the incorrect options should be derived by providing similar entities of the same taxonomy. 

3. Answer
- The correct answer letter.

4. Difficulty
- Classify each question into one of following difficulty levels: 1, 2, 3, 4.
- Difficulty Levels:
    - 1: Basic Recall - Simple facts explicitly stated on the page.
    - 2: Understanding - Slight interpretation or paraphrasing of information from the page.
    - 3: Applied Reasoning - Requires connecting multiple details or inferring something not stated word-for-word.
    - 4: Complex Insight - Deep reasoning, multi-step logic, or synthesizing several parts of the page to answer.

5. Rationale
-  A detailed explanation of why the correct answer is correct and why each incorrect option is
wrong.
- DO NOT reference the original source material in any part of the rationale.

Important:
- Questions must be self-contained, including any necessary context or definitions.
- Do not reference the original source material in any part of the question, options, or rationale.
- Ensure that only one option is unambiguously correct.

Format your QA pairs as an array of JSON objectsin the following JSON format, here are examples:

[
    {
        "question": "What was Harry Potter's home address?",
        "options": {
            "A": "4 Privet Drive, Little Whinging",
            "B": "4 Pivet Lane, Little Whimbrel",
            "C": "14 Private Drive, Little Whitting",
            "D": "4 Privy Court, Little Windham"
        },
        "answer": "A",
        "difficulty": 1,
        "rationale": "Option A is correct because it specifies the exact street name and town associated with Harry's residence. Option B is incorrect because although it has a similar structure, both the street name and town are altered. Option C is wrong because the street name is slightly modified and the house number is incorrect. Option D is incorrect because both the street and town names differ from the correct location. "
    },
    {
        "question": "What physical characteristic most clearly distinguishes a Thestral from other winged magical creatures, such as Hippogriffs or Winged Horses?",
        "options": {
            "A": "They have shimmering scales and a long, barbed tail used for defense.",
            "B": "They possess feathered wings and bright silver fur.",
            "C": "They have white, shining eyes and skin that clings directly to their bones.",
            "D": "They are covered in thick black feathers and have glowing red eyes."
        },
        "answer": "C",
        "difficulty": 2,
        "rationale": "Option C is correct because it accurately describes the distinctive skeletal appearance and white, shining eyes of Thestrals. Option A is incorrect because Thestrals do not have shimmering scales or a barbed tail. Option B is incorrect because Thestrals do not have feathers or silver fur. Option D is incorrect because they do not possess feathers or red eyes."
    }
]

"""

def group_documents_by_page():
    vectorstore = Chroma(
        collection_name="harry_potter_collection", 
        persist_directory=CHROMA_PATH, 
        embedding_function=OpenAIEmbeddings()
    )

    raw = vectorstore.get()
    documents = raw['documents']
    metadatas = raw['metadatas']

    from collections import defaultdict
    page_groups = defaultdict(str)

    # n = len(documents)
    n = 3

    for i in range(n):
        cur_document = documents[i]
        cur_metadata = metadatas[i]
        title = cur_metadata['source'].split('hp_data/')[1].split('.txt')[0]
        page_groups[title] += cur_document 

    return page_groups

def generate_questions(page_groups):
    dataset = []
    for title, document in page_groups.items():
        # have LLM automatically generate 5 questions for each document
        chat_input = [
            {
                "role": "system",
                "content": QA_GENERATION_PROMPT
            },
            {
                "role": "user",
                "content": f"Title: {title}\nDocument: {document}"
            }
        ]
        llm_instance = get_llm()

        response = generate_text(chat_input, llm_instance)

        # parse response into list of JSON objects
        try:
            response_json = json.loads(response.strip())
        except json.JSONDecodeError:
            print(f"Error parsing response: {response}")
            continue
        
        for qa in response_json:
            dataset.append(qa)

    return dataset

def save_dataset(dataset):
    df = pd.json_normalize(dataset)
    df.to_csv("qa_dataset.csv", index=False)

if __name__ == "__main__":
    page_groups = group_documents_by_page()
    dataset = generate_questions(page_groups)
    save_dataset(dataset)
