from .host_agent import HostAgent
from langchain_ollama import ChatOllama


root_agent = HostAgent(['http://localhost:10000']).create_agent()

# Initialize the Ollama chat LLM
ollama_chat_llm = ChatOllama(
    base_url="http://127.0.0.1:11434",
    model="llama3.2:latest",
    temperature=0.2
)

def get_ollama_response(prompt: str) -> str:
    """
    Sends the prompt to the local Ollama model and returns the response as a string.
    """
    # The simplest way: just call the LLM with the prompt
    result = ollama_chat_llm.invoke(prompt)
    return str(result)
