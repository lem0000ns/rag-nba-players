from dotenv import load_dotenv
import argparse
from rag_chain import get_qa_chain
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
import re
from langchain_openai import ChatOpenAI

load_dotenv()

rag_chain = get_qa_chain()

def rag_tool_func(q):
    # print(f"\nRAG tool called with question: {q}")
    result = rag_chain.invoke({"input": q})
    # print("-" * 100)
    # print(f"\nRetrieved {len(result['context'])} documents:")
    # for i, doc in enumerate(result['context'], 1):
    #     print(f"\n--- Document {i} ---")
    #     print(f"Content: {doc.page_content}")
    #     print(f"Metadata: {doc.metadata}")
    # print("-" * 100)
    answer = result.get("answer", "No answer found")
    return answer

tools = [
    Tool(
        name="RAG",
        func=rag_tool_func,
        description=f"Use this FIRST to answer questions about any Harry Potter related lore, including characters, magic, places, and events."
    )
]

REACT_PROMPT = """You are a helpful assistant that answers questions about Harry Potter related lore. You have access to the following tools:
{tools}

IMPORTANT INSTRUCTIONS:
1. ALWAYS try the RAG tool for any question about Harry Potter related lore
2. Once a final answer is found, STOP and provide the Final Answer immediately

To use a tool, you MUST use the following format:
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the following format:
Thought: Do I need to use a tool? No
Final Answer: [your response here]

It's very important to always include the 'Thought' before any 'Action' or 'Final Answer'. Ensure your output strictly follows the formats above.

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
"""


def get_agent_chain():
    prompt = PromptTemplate.from_template(REACT_PROMPT)
    agent = create_react_agent(llm=ChatOpenAI(model="gpt-4-turbo"), tools=tools, prompt=prompt)
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True, output_key="output")
    agent_chain = AgentExecutor(agent=agent,
                            tools=tools,
                            memory=memory,
                            max_iterations=5,
                            handle_parsing_errors=True,
                            verbose=False,
                            )
    return agent_chain

def sanitize_query(query):
    query = query.strip().lower() # remove leading and trailing spaces and convert to lowercase
    query = re.sub(r'[^\w\s?!]', '', query) # remove special characters
    return " ".join(query.split()) # remove extra spaces

def main():
    agent_chain = get_agent_chain()
    while True:
        user_query = sanitize_query(input("> "))
        if user_query == "q":
            break
        # use agent chain to simulate conversation
        response = agent_chain.invoke({"input": user_query})
        print(response['output'])
    print("Goodbye!")

if __name__ == "__main__":
    main()