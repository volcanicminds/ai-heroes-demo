from collections.abc import AsyncIterable
from typing import Any, Literal
import logging
import math
import re

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
    """Use this to perform mathematical calculations.
    
    Args:
        expression: A string representing a mathematical expression to evaluate.
                   Must be a valid Python expression using only allowed functions.
                   Examples:
                   - "12 * 12"
                   - "sqrt(16)"
                   - "sin(pi/2)"
                   - "sqrt((12 * 12) / 78)"
                   
    Returns:
        A dictionary containing the formatted result or error message.
    """
    logger.info(f"[Calculator] Received expression to evaluate: {expression}")
    
    # Clean up the input expression
    expression = expression.strip()
    
    # Handle natural language expressions
    expression = (expression.lower()
                 .replace('calculate', '')
                 .replace('what is', '')
                 .replace('the square root of', 'sqrt')
                 .replace('square root', 'sqrt')
                 .replace('the result of', '')
                 .strip())

    # Remove extra spaces around operators
    expression = re.sub(r'\s*([+\-*/])\s*', r'\1', expression)
    
    # Handle "sqrt" at the end of expression
    if expression.endswith('sqrt'):
        expression = 'sqrt(' + expression[:-4].strip() + ')'
    # Handle "sqrt" anywhere else
    elif 'sqrt' in expression and not 'sqrt(' in expression:
        expression = f"sqrt({expression.replace('sqrt', '')})"
    
    # Sanitize the input by removing any unexpected characters
    valid_chars = set('0123456789.()*/+-[] abcdefghijklmnopqrstuvwxyz')
    if not all(c.lower() in valid_chars for c in expression):
        error_msg = f'Invalid characters in expression: {expression}'
        logger.error(f"[Calculator] {error_msg}")
        return {'error': error_msg, 'expression': expression}
    
    try:
        # Create a safe dict with only math functions we want to allow
        safe_dict = {
            'abs': abs,
            'round': round,
            'pow': pow,
            'sum': sum,
            'max': max,
            'min': min,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e
        }
        
        # Log available functions for debugging
        logger.debug(f"[Calculator] Available functions: {list(safe_dict.keys())}")
        
        # Evaluate the expression in a restricted environment
        logger.info(f"[Calculator] Evaluating expression: '{expression}' with safe_dict...")
        result = eval(expression.strip(), {"__builtins__": {}}, safe_dict)
        logger.info(f"[Calculator] Evaluation successful, result: {result}")
        
        # Format the result to a reasonable number of decimal places if it's a float
        if isinstance(result, float):
            result = round(result, 6)
            
        return {'result': result, 'expression': expression}
    except Exception as e:
        error_msg = f'Calculation error: {str(e)}'
        logger.error(f"[Calculator] {error_msg}")
        return {'error': error_msg, 'expression': expression}


class ResponseFormat(BaseModel):
    """Response format that ensures only numeric results are returned."""

    status: Literal['input_required', 'completed', 'error'] = 'completed'
    message: str
    
    @classmethod
    def from_calculation(cls, calc_result: dict) -> 'ResponseFormat':
        """Create a response containing only the numeric result."""
        logger.info(f"[ResponseFormat] Processing calculation result: {calc_result}")
        
        if 'error' in calc_result:
            logger.error(f"[ResponseFormat] Error in calculation: {calc_result['error']}")
            return cls(
                status='error',
                message=calc_result['error']
            )
        
        # Extract the numeric result
        result = calc_result['result']
        logger.debug(f"[ResponseFormat] Raw result type: {type(result)}, value: {result}")
        
        # Format numeric values consistently
        if isinstance(result, (int, float)):
            # Format floats with 6 decimal places, strip trailing zeros
            formatted_result = f"{float(result):.6f}".rstrip('0').rstrip('.')
            logger.debug(f"[ResponseFormat] Formatted result: {formatted_result}")
            # Extract just the number from any potential text
            numeric_match = re.search(r'-?\d*\.?\d+(?:e[-+]?\d+)?', formatted_result)
            if numeric_match:
                numeric_value = numeric_match.group()
                return cls(
                    status='completed',
                    message=numeric_value
                )
            else:
                logger.error(f"[ResponseFormat] No numeric value found in result")
                return cls(
                    status='error',
                    message='Invalid numeric format'
                )
        else:
            logger.error(f"[ResponseFormat] Invalid result type: {type(result)}")
            return cls(
                status='error',
                message='Invalid result type'
            )

    def model_dump(self) -> dict:
        """Return the response data with properly formatted numeric values."""
        logger.debug(f"[ResponseFormat] model_dump called on message: {self.message}")
        data = super().model_dump()
        
        # No additional formatting needed since from_calculation already formats correctly
        logger.debug(f"[ResponseFormat] Final output: {data}")
        return data


class CalculationAgent:
    SYSTEM_INSTRUCTION = """You are a calculator agent that MUST follow these rules EXACTLY:

1. ALWAYS use the calculate() tool for ALL calculations
2. Return ONLY THE NUMBER - nothing else
3. NO words before or after the number
4. NO explanations
5. NO units
6. NO "The answer is" or similar phrases
7. For complex expressions, combine them into a SINGLE calculate() call

Input: "what is 5 plus 3"
Response: 8

Input: "calculate the square root of 16"
Response: 4

Input: "what is (10 * 5) / 2"
Response: 25

Input: "tell me the result of sin(pi/2)"
Response: 1

Input: "what's sqrt((12 * 12) / 78)"
Response: 1.358732

CRITICAL: Your response must contain ONLY the number - no other characters or text."""

    def __init__(self):
         # Use Ollama's llama2 model via LangChain Ollama integration
        logger.info("[Calculator Agent] Initializing with model acidtib/qwen2.5-coder-cline:7b")
        self.model = ChatOllama(
            model="llama3.2:latest", 
            temperature=0,
            system=self.SYSTEM_INSTRUCTION  # Explicitly set system message
        )
        self.tools = [calculate]

        # Log system instruction to verify it's being used
        logger.info(f"[Calculator Agent] Using system instruction:\n{self.SYSTEM_INSTRUCTION}")

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
        )
        logger.info("[Calculator Agent] Initialization complete")

    def invoke(self, query, sessionId) -> str:
        logger.info(f"[Calculator Agent] Processing query: {query} (session: {sessionId})")
        config = {'configurable': {'thread_id': sessionId}}
        response = self.graph.invoke({'messages': [('user', query)]}, config)
        logger.info(f"[Calculator Agent] Query response: {response}")
        return self.get_agent_response(config)

    async def stream(self, query, sessionId) -> AsyncIterable[dict[str, Any]]:
        logger.info(f"[Calculator Agent] Starting stream for query: {query} (session: {sessionId})")
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': sessionId}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                logger.info("[Calculator Agent] Processing tool call...")
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Computing the result...',
                }
            elif isinstance(message, ToolMessage):
                logger.info("[Calculator Agent] Tool message received...")
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the calculation..',
                }

        final_response = self.get_agent_response(config)
        logger.info(f"[Calculator Agent] Stream complete, final response: {final_response}")
        yield final_response

    def get_agent_response(self, config):
        logger.info("[Calculator Agent] Getting agent response...")
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        logger.info(f"[Calculator Agent] Structured response: {structured_response}")
        
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            response = {
                'is_task_complete': structured_response.status == 'completed',
                'require_user_input': structured_response.status in ['input_required', 'error'],
                'content': structured_response.message,
            }
            logger.info(f"[Calculator Agent] Formatted response: {response}")
            return response

        error_msg = 'We are unable to process your request at the moment. Please try again.'
        logger.error(f"[Calculator Agent] Error getting response: {error_msg}")
        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': error_msg,
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
