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
    """Routes the message to the selected agent and returns its response."""
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
            ("system", """You are a routing assistant. Your job is to:
1. Use the `discover_agents` tool to find available agents.
2. Based on the user's message, choose the most appropriate agent.
3. Use the `route_message` tool to send the message to the selected agent.
Always start by discovering agents unless the message specifically mentions that step."""),
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
        try:
            logger.info(f"Invoking with query: {query}")
            return asyncio.run(self.async_invoke(query, session_id))
        except RuntimeError as e:
            # Fallback per event loop già attivo (es. in FastAPI)
            logger.warning("Detected running event loop, using fallback.")
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.create_task(self.async_invoke(query, session_id))
            else:
                return loop.run_until_complete(self.async_invoke(query, session_id))

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        logger.warning("Streaming is not supported.")
        raise NotImplementedError("Streaming is not implemented.")