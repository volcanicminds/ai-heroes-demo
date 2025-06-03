import logging
import uuid
from typing import List, Dict, Any, Union
import asyncio
from collections.abc import AsyncIterable
import httpx
import os
from dotenv import load_dotenv

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from common.client.card_resolver import A2ACardResolver
from common.client.client import A2AClient
from common.types import TaskSendParams, Message, TextPart

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain").setLevel(logging.WARNING)
logger = logging.getLogger("agent")

AGENT_URLS = ["http://localhost:10000", "http://localhost:10001"]
DEFAULT_TIMEOUT: Union[float, tuple[float, float, float, float], None] = 180.0  # 3 minutes timeout

@tool
def discover_agents() -> List[Dict[str, Any]]:
    """Discovers other A2A agents at predefined URLs."""
    logger.info("🔍 Discovering available agents...")
    discovered_agents = []
    for url in AGENT_URLS:
        try:
            card = A2ACardResolver(url).get_agent_card()
            discovered_agents.append({
                "name": card.name,
                "description": card.description,
                "url": url
            })
            logger.info(f"✅ Found available agent '{card.name}' running at {url}")
        except Exception as e:
            logger.warning(f"❌ Could not reach agent at {url}: {e}")
    return discovered_agents

@tool
async def route_message(agent_url: str, message: str, session_id: str) -> str:
    """
    Routes a message to another agent and returns their response.
    Args:
        agent_url: The URL of the target agent (e.g., http://localhost:10000)
        message: The message to send
        session_id: The session ID for tracking the conversation
    """
    try:
        logger.info(f"🔗 Routing message to agent at {agent_url} with session_id: {session_id}")
        client = A2AClient(url=agent_url, timeout=DEFAULT_TIMEOUT)
        task_id = str(uuid.uuid4())
        request = TaskSendParams(
            id=task_id,
            sessionId=session_id,
            message=Message(role="user", parts=[TextPart(text=message)]),
            acceptedOutputModes=["text", "text/plain"]
        )
        logger.info(f"🚀 Forwarding request to agent at {agent_url} (timeout: {DEFAULT_TIMEOUT}s)")
        
        try:
            response = await client.send_task(request)
        except httpx.ReadTimeout:
            error_msg = f"⏰ Request timed out after {DEFAULT_TIMEOUT} seconds waiting for {agent_url}"
            logger.error(error_msg)
            return f"{error_msg}. The agent might be busy or not responding. Try increasing the timeout or try again later."
        except httpx.ConnectError:
            error_msg = f"❌ Connection failed to {agent_url}"
            logger.error(error_msg)
            return f"{error_msg}. Please check if the agent is running and accessible."
        except httpx.HTTPError as e:
            error_msg = f"HTTP error occurred while connecting to {agent_url}: {e}"
            logger.error(error_msg)
            return f"Error communicating with agent: {str(e)}"
        
        # Enhanced response handling
        if response and hasattr(response, 'result'):
            result = response.result
            
            # Check if there's a status with a message
            if hasattr(result, 'status') and hasattr(result.status, 'message'):
                if hasattr(result.status.message, 'parts'):
                    for part in result.status.message.parts:
                        if hasattr(part, 'text') and part.text:
                            return part.text
            
            # Fallback to checking result content
            if hasattr(result, 'content'):
                return result.content
            
            # Check artifacts as last resort
            if hasattr(result, 'artifacts') and result.artifacts:
                for artifact in result.artifacts:
                    if hasattr(artifact, 'parts') and artifact.parts:
                        for part in artifact.parts:
                            if hasattr(part, 'text') and part.text:
                                return part.text
            
            logger.warning(f"⚠️ Unexpected response format received: {response}")
            return "Unable to extract response from agent"
            
        return "No response received from agent"
    except Exception as e:
        import traceback
        logger.error(f"Failed to route message: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return f"Error processing agent response: {str(e)}. Check logs for details."

class LangchainAgent:
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        # Use Google Gemini model instead of Ollama
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",  # or another Gemini model name if needed
            temperature=0,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        self.tools = [discover_agents, route_message]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a message routing assistant that intelligently forwards messages to the most appropriate agent based on their capabilities."),
            ("system", """Steps:
                1. Call discover_agents *tool* to get available agents
                2. Choose the most appropriate agent by matching the task to agent capabilities:
                   - A Content Generation Agentic System: For writing articles, stories, explanations
                   - A Calculator Agent: For math calculations only
                3. Call route_message *tool* with:
                   - agent_url from discover_agents results only
                   - message unchanged
                   - session_id exactly as provided"""),
            ("human", "{message}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_tool_calling_agent(
            llm=self.model,
            tools=self.tools,
            prompt=self.prompt
        )

        self.runnable = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

    async def async_invoke(self, query: str, session_id: str) -> Dict[str, Any]:
        try:
            logger.info(f"🤖 Starting agent with query: '{query}' with session_id: {session_id}")
            # Reinitialize the model and agent to ensure a fresh connection
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-04-17",
                temperature=0,
                google_api_key=os.environ.get("GOOGLE_API_KEY")
            )
            # Recreate the agent with the new model
            self.agent = create_tool_calling_agent(
                llm=self.model,
                tools=self.tools,
                prompt=self.prompt
            )
            # Update the runnable with the new agent
            self.runnable = AgentExecutor.from_agent_and_tools(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
            )
            # Pass a single input dict
            output = await self.runnable.ainvoke({
                "message": query,
                "session_id": session_id
            })

            logger.info("✅ Agent execution completed successfully with output:")
            
            # Enhanced response handling
            if isinstance(output, dict):
                steps = output.get('intermediate_steps', [])
                logger.info(f"🔍 Found {len(steps)} intermediate steps in the output")
                for step in reversed(steps):
                    if isinstance(step, tuple) and len(step) == 2:
                        action, result = step
                        if action.tool == 'route_message':
                            logger.info(f"✨ Response received: {result}")
                            if isinstance(result, str) and not result.startswith("Routing failed"):
                                return {
                                    "is_task_complete": True,
                                    "require_user_input": False,
                                    "content": result
                                }
                
                # If we got here, no valid route_message response was found
                logger.warning("❌ No valid response received from target agent")
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": "Failed to get response from target agent."
                }
            
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": "Failed to process agent response."
            }
        except Exception as e:
            logger.exception(f"❌ Agent execution failed: {e}")
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Error during processing: {str(e)}"
            }

    def invoke(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for async_invoke.
        This method ensures we properly handle the async/sync boundary.
        """
        try:
            logger.info(f"🎯 Processing request: '{query}'")
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("⚡️ Event loop already active - switching to thread execution")
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(self.async_invoke(query, session_id)))
                    return future.result()
            else:
                return loop.run_until_complete(self.async_invoke(query, session_id))
        except Exception as e:
            logger.exception(f"Error in invoke: {e}")
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"An error occurred during processing: {str(e)}"
            }

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        logger.warning("⚠️ Streaming functionality is not yet implemented")
        raise NotImplementedError("Streaming is not implemented.")
