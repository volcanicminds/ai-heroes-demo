import logging
import uuid
from typing import List, Dict, Any, Union
import asyncio
from collections.abc import AsyncIterable
import httpx

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
DEFAULT_TIMEOUT: Union[float, tuple[float, float, float, float], None] = 120.0  # 2 minutes timeout

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
        self.model = ChatOllama(model="acidtib/qwen2.5-coder-cline:7b", temperature=0)
        self.tools = [discover_agents, route_message]

        self.prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a specialized routing assistant. Your ONLY purpose is to forward messages to other agents and return their responses.
        MANDATORY STEPS (execute in order):
        1. Call `discover_agents()` to get the list of available agents
        2. Select the most appropriate agent URL based on the request
        3. Use `route_message(agent_url, message, session_id)` to forward the request and return its response

        IMPORTANT:
        - The session_id is provided in the input variables
        - You MUST use the exact session_id value provided
        - You MUST return only the response from route_message
         
        CRITICAL: Always call `discover_agents` tool first to get the list of available agents.

        Example usage [THIS IS JUST AND EXAMPLE]:
        Input: "Get USD/EUR rate"
        1. discover_agents() <--- This MUST be called first ALWAYS
        2. route_message(
            agent_url="http://localhost:10000", [THIS IS JUST AND EXAMPLE]
            message="Get USD/EUR rate", 
            session_id=input_variables.session_id  # Use the provided session_id
        )
         CRITICAL: session_id is available directly in the context. DO NOT use 'input_variables.session_id'
         """),
        ("human", "{message}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("system", "Remember: You MUST call both discover_agents and route_message tools, in that order.")
    ])

        agent = create_tool_calling_agent(
            llm=self.model,
            tools=self.tools,
            prompt=self.prompt
        )

        self.runnable = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            return_intermediate_steps=True
            )

    async def async_invoke(self, query: str, session_id: str) -> Dict[str, Any]:
        try:
            logger.info(f"🤖 Starting agent with query: '{query}'")
            # Reinitialize the model and agent to ensure a fresh connection
            self.model = ChatOllama(model="acidtib/qwen2.5-coder-cline:7b", temperature=0)
            # Recreate the agent with the new model
            agent = create_tool_calling_agent(
                llm=self.model,
                tools=self.tools,
                prompt=self.prompt
            )
            # Update the runnable with the new agent
            self.runnable = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,
                return_intermediate_steps=True
            )
            # Make session_id directly available in the context
            output = await self.runnable.ainvoke({
                "message": query,
                "session_id": session_id
            })
            
            # Enhanced response handling
            if isinstance(output, dict):
                steps = output.get('intermediate_steps', [])
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
            # Use asyncio.get_event_loop().run_until_complete for running the async function
            # from synchronous code
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("⚡️ Event loop already active - switching to thread execution")
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
        logger.warning("⚠️ Streaming functionality is not yet implemented")
        raise NotImplementedError("Streaming is not implemented.")
