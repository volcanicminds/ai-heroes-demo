"""Langchain based sample for A2A protocol.

Handles the agents and also presents the tools required.
"""

import logging
from typing import List, Dict, Any

import httpx

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage
from collections.abc import AsyncIterable
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor

from common.client.card_resolver import A2ACardResolver
from common.client.client import A2AClient
from common.types import AgentCard, TaskSendParams, Message, TextPart, TaskState, TaskStatus, Task, Artifact, SendTaskResponse

import uuid # Import uuid for generating task IDs
from typing import List, Dict, Any

from common.client.card_resolver import A2ACardResolver
from common.client.client import A2AClient
from common.types import AgentCard, TaskSendParams, Message, TextPart, TaskState, TaskStatus, Task, Artifact, SendTaskResponse

logging.basicConfig(level=logging.DEBUG)  # Or use INFO to reduce verbosity
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)
logging.getLogger("langchain_community").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# URLs of the agents to discover
AGENT_URLS = ["http://localhost:10000", "http://localhost:10001"]

@tool
def discover_agents() -> List[Dict[str, Any]]:
    """Discovers other Google A2A agents at predefined URLs and returns their names and descriptions."""
    logger.info("Attempting to discover agents.")
    discovered_agents = []
    for url in AGENT_URLS:
        try:
            card_resolver = A2ACardResolver(url)
            card = card_resolver.get_agent_card()
            discovered_agents.append({
                "name": card.name,
                "description": card.description,
                "url": url # Include the URL for later use
            })
            logger.info(f"Discovered agent: {card.name} at {url}")
        except httpx.HTTPStatusError as e:
            logger.warning(f"Could not discover agent at {url}: HTTP error {e.response.status_code}")
        except httpx.RequestError as e:
            logger.warning(f"Could not discover agent at {url}: Request error {e}")
        except Exception as e:
            logger.warning(f"Could not discover agent at {url}: An unexpected error occurred: {e}")
    logger.info(f"Finished agent discovery. Found {len(discovered_agents)} agents.")
    return discovered_agents

async def send_message_to_agent(agent_url: str, message_content: str, session_id: str) -> SendTaskResponse:
    """Sends a message to a specific agent."""
    logger.info(f"Attempting to send message to agent at {agent_url}.")
    try:
        client = A2AClient(agent_url)
        task_id = str(uuid.uuid4()) # Generate a unique task ID
        request = TaskSendParams(
            id=task_id,
            sessionId=session_id,
            message=Message(
                role='user',
                parts=[TextPart(text=message_content)],
            ),
            acceptedOutputModes=['text', 'text/plain'], # Specify accepted output modes
        )
        logger.info(f"Sending task to {agent_url} with task_id: {task_id}")
        response = await client.send_task(request)
        logger.info(f"Received response from {agent_url} for task_id {task_id}: {response}")
        return response
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to send message to {agent_url}: HTTP error {e.response.status_code}")
        raise
    except httpx.RequestError as e:
        logger.error(f"Failed to send message to {agent_url}: Request error {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to send message to {agent_url}: An unexpected error occurred: {e}")
        raise

@tool
def route_message(agent_url: str, message: str, session_id: str) -> str:
    """Routes the message to the selected agent via Google A2A and returns the response."""
    import asyncio
    response = asyncio.run(send_message_to_agent(agent_url, message, session_id))
    return response.task.outputs[0].text if response.task.outputs else "No response from agent"

class LangchainAgent:
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        self.model = ChatOllama(model="orieg/gemma3-tools:4b", temperature=0)
        logger.info("Initializing LangchainAgent.")
        self.tools = [discover_agents, route_message]
        self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a routing assistant. Your job is to:
                            1. Use the `discover_agents` tool to find agents.
                            2. Decide which agent is best for the user's message.
                            3. Use the `route_message` tool with the selected agent's URL.
                            Respond with the final result from the routed agent.

                            You MUST use the `discover_agents` tool to find other agents BEFORE considering the user message.
                            This is something you MUST do at every execution event if the user message has not "discover agents" anywhere.
                            After that, you MUST use the `route_message` tool to send the message to the selected agent.
                """),
                ("human", "{message}"),
            ])

        # ✅ Create an actual agent (adds planning + tool calling logic)
        agent = create_tool_calling_agent(llm=self.model, tools=self.tools, prompt=self.prompt)

        # ✅ Use AgentExecutor to handle looped reasoning & tool invocation
        self.runnable: Runnable = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        logger.info("LangchainAgent initialized.")

    def process_message(self, message_content: str, session_id: str) -> str:
        logger.info(f"Processing message: '{message_content}' for session: {session_id}")

        # Let the LLM + tools handle everything
        output = self.runnable.invoke({"message": message_content})
        
        if hasattr(output, 'content'):
            return output.content
        elif isinstance(output, str):
            return output
        else:
            return str(output)

    def invoke(self, query: str, session_id: str) -> dict:
        logger.info(f"Invoking LangchainAgent with query: {query}, sessionId: {session_id}")

        try:
            logger.debug(f"Running self.runnable.invoke with input: {query}")
            output = self.runnable.invoke({"message": query})
            logger.debug(f"LLM returned: {output}")
            
            content = output.content if hasattr(output, 'content') else str(output)

            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': content,
            }
        except Exception as e:
            logger.exception(f"LangchainAgent failed to process message: {e}")
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': "An internal error occurred while routing your message.",
            }

    async def stream(self, query, sessionId) -> AsyncIterable[dict[str, Any]]:
        """Streaming is not implemented for this agent."""
        logger.info("Attempting to stream, but streaming is not implemented.")
        raise NotImplementedError('Streaming is not supported by this agent.')
