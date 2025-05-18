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
    SYSTEM_INSTRUCTION = '''You are a calculator. OUTPUT ONLY THE calculate() FUNCTION CALL.

Available: abs(), sqrt(), sin(), cos(), tan(), pi, e

Examples - COPY THIS FORMAT EXACTLY:

Input: calculate 12 plus 78 times 123 and do the square root
Output: calculate("sqrt(12 + 78 * 123)")

Input: 21 * 122
Output: calculate("21 * 122")

Input: what's nine plus ten
Output: calculate("9 + 10")

Input: square root of 25 times 4
Output: calculate("sqrt(25 * 4)")

Input: sine of pi divided by 2
Output: calculate("sin(pi/2)")

REQUIRED FORMAT:
1. No text before or after calculate()
2. No explanations
3. No formatting
4. No "the answer is"
5. ONLY the calculate() call'''

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
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
        )

    def invoke(self, query: str, sessionId: str) -> dict:
        logger.info(f"[CalculationAgent] Invoking with query: {query}")
        config = {'configurable': {'thread_id': sessionId}}
        response = self.graph.invoke({'messages': [('user', query)]}, config)
        logger.info(f"[CalculationAgent] Graph invoke response: {response}")
        result = self.get_agent_response(config)
        logger.info(f"[CalculationAgent] Final response: {result}")
        return result

    async def stream(self, query: str, sessionId: str) -> AsyncIterable[dict[str, Any]]:
        logger.info(f"[CalculationAgent] Starting stream with query: {query}")
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': sessionId}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            logger.info(f"[CalculationAgent] Stream message: {message}")
            if isinstance(message, (AIMessage, ToolMessage)):
                logger.info(f"[CalculationAgent] Processing message type: {type(message)}")
                logger.info(f"[CalculationAgent] Message content: {message.content if hasattr(message, 'content') else None}")
                logger.info(f"[CalculationAgent] Tool calls: {message.tool_calls if hasattr(message, 'tool_calls') else None}")
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Computing...',
                }

        final_response = self.get_agent_response(config)
        logger.info(f"[CalculationAgent] Stream final response: {final_response}")
        yield final_response

    def get_agent_response(self, config: dict) -> dict:
        logger.info("[CalculationAgent] Getting agent response")
        current_state = self.graph.get_state(config)
        logger.info(f"[CalculationAgent] Current state: {current_state}")
        structured_response = current_state.values.get('structured_response')
        logger.info(f"[CalculationAgent] Structured response: {structured_response}")
        
        if isinstance(structured_response, ResponseFormat):
            logger.info(f"[CalculationAgent] Valid ResponseFormat received with status: {structured_response.status}")
            return {
                'is_task_complete': structured_response.status == 'completed',
                'require_user_input': structured_response.status in ['input_required', 'error'],
                'content': structured_response.message,
            }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'Unable to process request. Please try again.',
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
