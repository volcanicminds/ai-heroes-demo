from collections.abc import AsyncIterable
from typing import Any, Literal
import logging
import math

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)
memory = MemorySaver()

@tool
def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression.
    Args:
        expression: A mathematical expression (e.g. "12 * 12", "sqrt(16)", "sin(pi/2)")
    Returns:
        Dictionary with result or error message.
    """
    if not isinstance(expression, str):
        return {'error': 'Expression must be a string', 'expression': str(expression)}
        
    try:
        expr = expression.strip()
        safe_dict = {
            'abs': abs,
            'sqrt': math.sqrt, 
            'sin': math.sin, 
            'cos': math.cos, 
            'tan': math.tan,
            'pi': math.pi, 
            'e': math.e
        }
        
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return {'result': float(result), 'expression': expr}
    except Exception as e:
        return {'error': str(e), 'expression': expression}


class ResponseFormat(BaseModel):
    """Response format for calculation results."""
    status: Literal['input_required', 'completed', 'error'] = 'completed'
    message: str
    
    @classmethod
    def from_calculation(cls, calc_result: dict) -> 'ResponseFormat':
        """Create a response containing only the numeric result."""
        if 'error' in calc_result:
            return cls(status='error', message=calc_result['error'])
        
        result = calc_result['result']
        if isinstance(result, (int, float)):
            # Format integers as is, floats with up to 6 decimals
            formatted = f"{float(result):.6f}".rstrip('0').rstrip('.') if isinstance(result, float) else str(result)
            return cls(status='completed', message=formatted)
        
        return cls(status='error', message='Invalid result type')


class CalculationAgent:
    SYSTEM_INSTRUCTION = '''You are a calculator that processes mathematical queries. Output must ONLY be function calls in correct format.

Available Math Operations:
- Basic: +, -, *, /, ()
- Functions: abs(), sqrt(), sin(), cos(), tan()
- Constants: pi, e

EXAMPLES OF VALID RESPONSES:

User: calculate square root of 144
Assistant: calculate("sqrt(144)")

User: what is 3 plus 4
Assistant: calculate("3 + 4")

User: sine of pi
Assistant: calculate("sin(pi)")

User: what is five plus ten
Assistant: calculate("5 + 10")

STRICT REQUIREMENTS:
1. ONLY output the calculate() function call
2. Expression MUST be in double quotes
3. Use mathematical operators (+,-,*,/)
4. No explanations or extra text
5. No formatting or whitespace
6. First parse query to math expression
7. Then wrap in calculate("...")'''

    def __init__(self):
        self.model = ChatOllama(
            model="llama3.2:latest", 
            temperature=0,
            system=self.SYSTEM_INSTRUCTION
        )
        self.graph = create_react_agent(
            self.model,
            tools=[calculate],
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION
        )

    def invoke(self, query: str, sessionId: str) -> dict:
        logger.info(f"🤔 Processing calculation request: '{query}'")
        config = {'configurable': {'thread_id': sessionId}}
        response = self.graph.invoke({'messages': [('user', query)]}, config)
        logger.info(f"⚡️ Starting calculation pipeline...")
        result = self.get_agent_response(config)
        logger.info(f"✨ Result: {result['content']}")
        return result

    async def stream(self, query: str, sessionId: str) -> AsyncIterable[dict[str, Any]]:
        logger.info(f"🚀 Starting interactive calculation session")
        logger.info(f"📝 User Query: '{query}'")
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': sessionId}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if isinstance(message, (AIMessage, ToolMessage)):
                logger.info(f"🧮 Processing mathematical expression...")
                if hasattr(message, 'content'):
                    logger.info(f"💭 AI thinking process: {message.content}")
                if hasattr(message, 'tool_calls'):
                    logger.info(f"🛠️ Mathematical tools in use: {message.tool_calls}")
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Computing...',
                }

        final_response = self.get_agent_response(config)
        logger.info(f"🎉 Calculation complete! Answer: {final_response['content']}")
        yield final_response

    def get_agent_response(self, config: dict) -> dict:
        logger.info("📊 Preparing final result...")
        current_state = self.graph.get_state(config)
        logger.info(f"🔄 Current state: Processing complete")
        
        # Get the last message from the state
        messages = current_state.values.get('messages', [])
        if not messages:
            logger.warning("❌ No response received from calculation engine")
            return {
                'is_task_complete': False,
                'require_user_input': True,
                'content': 'No response received',
            }
            
        # Find the last result from a calculate tool call
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name == 'calculate':
                try:
                    result = eval(msg.content)  # Safe since we know this is our JSON response
                    if 'error' in result:
                        logger.error(f"⚠️ Calculation error: {result['error']}")
                        return {
                            'is_task_complete': False,
                            'require_user_input': True,
                            'content': result['error'],
                        }
                    logger.info(f"✅ Calculation successful!")
                    return {
                        'is_task_complete': True,
                        'require_user_input': False,
                        'content': str(result['result']),
                    }
                except Exception as e:
                    logger.error(f"❌ Error processing result: {str(e)}")
                break

        logger.warning("⚠️ Unable to process calculation request")
        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'Unable to process request. Please try again.',
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
