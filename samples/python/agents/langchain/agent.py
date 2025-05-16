import logging
import uuid
from typing import List, Dict, Any
import asyncio
from collections.abc import AsyncIterable

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from common.client.card_resolver import A2ACardResolver
from common.client.client import A2AClient
from common.types import TaskSendParams, Message, TextPart

# Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain").setLevel(logging.WARNING)
logger = logging.getLogger("agent")

AGENT_URLS = ["http://localhost:10000", "http://localhost:10001"]

@tool
def discover_agents() -> List[Dict[str, Any]]:
    """Discovers other A2A agents at predefined URLs."""
    discovered_agents = []
    for url in AGENT_URLS:
        try:
            card = A2ACardResolver(url).get_agent_card()
            discovered_agents.append({
                "name": card.name,
                "description": card.description,
                "url": url
            })
            logger.info(f"Discovered agent: {card.name} at {url}")
        except Exception as e:
            logger.warning(f"Failed to contact {url}: {e}")
    return discovered_agents

@tool
async def route_message(agent_url: str, message: str, session_id: str) -> str:
    """
    Sends the user's message to the specified agent URL and returns the agent's response.
    Use this tool after selecting the appropriate agent based on the user's query.
    """
    try:
        client = A2AClient(agent_url)
        task_id = str(uuid.uuid4())
        request = TaskSendParams(
            id=task_id,
            sessionId=session_id,
            message=Message(role="user", parts=[TextPart(text=message)]),
            acceptedOutputModes=["text", "text/plain"]
        )
        logger.info(f"Sending task to agent at {agent_url}")
        response = await client.send_task(request)
        return response.task.outputs[0].text if response.task.outputs else "No response"
    except Exception as e:
        logger.error(f"Failed to route message: {e}")
        return "Routing failed due to an error."

class LangchainAgent:
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        self.model = ChatOllama(model="orieg/gemma3-tools:4b", temperature=0)
        self.tools = [discover_agents, route_message]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a routing assistant that helps direct messages to the right specialized agent. Follow these steps exactly:

1. First, use the `discover_agents` tool to find available agents.
2. Analyze the list of agents and select the most appropriate one based on the user's request.
3. IMPORTANT: You MUST use the `route_message` tool to forward the user's message to the selected agent's URL.
4. Return only the response from the selected agent.

ALWAYS complete all steps and ensure you actually call route_message to forward the request.
DO NOT just say which agent you'll use - you must actually route the message.

Example flow:
1. Discover agents
2. Select appropriate agent
3. Use route_message(selected_agent_url, user_message, session_id)
4. Return the response"""),
            ("human", "{message}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_tool_calling_agent(
            llm=self.model,
            tools=self.tools,
            prompt=self.prompt
        )

        self.runnable = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    async def async_invoke(self, query: str, session_id: str) -> Dict[str, Any]:
        try:
            logger.info(f"Async invoking with query: {query}")
            output = await self.runnable.ainvoke({"message": query, "session_id": session_id})
            content = output.content if hasattr(output, "content") else str(output)
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": content
            }
        except Exception as e:
            logger.exception(f"Async invocation failed: {e}")
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": "An error occurred during processing."
            }

    def invoke(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for async_invoke.
        This method ensures we properly handle the async/sync boundary.
        """
        try:
            logger.info(f"Invoking with query: {query}")
            # Use asyncio.get_event_loop().run_until_complete for running the async function
            # from synchronous code
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("Event loop already running, cannot use run_until_complete. Using run in thread.")
                # If we're in an already running loop (e.g., inside FastAPI),
                # we can't run_until_complete - must use asyncio.run_coroutine_threadsafe
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(self.async_invoke(query, session_id)))
                    return future.result()
            else:
                # Normal case - run the coroutine in the current loop
                return loop.run_until_complete(self.async_invoke(query, session_id))
        except Exception as e:
            logger.exception(f"Error in invoke: {e}")
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"An error occurred during processing: {str(e)}"
            }

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        logger.warning("Streaming is not supported.")
        raise NotImplementedError("Streaming is not implemented.")
