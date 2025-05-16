from collections.abc import AsyncIterable
from typing import Any, Literal
import math
import logging

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
                   
    Returns:
        A dictionary containing the result or an error message.
    """
    logger.info(f"[Calculator] Received expression to evaluate: {expression}")
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
        logger.info(f"[Calculator] Evaluating expression with safe_dict...")
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        logger.info(f"[Calculator] Evaluation successful, result: {result}")
        return {'result': result}
    except Exception as e:
        error_msg = f'Calculation error: {str(e)}'
        logger.error(f"[Calculator] {error_msg}")
        return {'error': error_msg}


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class CurrencyAgent:
    SYSTEM_INSTRUCTION = (
        'You are a specialized calculator assistant. '
        "Your sole purpose is to use the 'calculate' tool to solve mathematical problems. "
        'IMPORTANT: You must format the input expression for the calculate tool as a valid Python expression. For example:\n'
        '- User asks: "what is 12 times 12?" → Use: calculate("12 * 12")\n'
        '- User asks: "calculate square root of 16" → Use: calculate("sqrt(16)")\n'
        '- User asks: "what is sine of pi/2?" → Use: calculate("sin(pi/2)")\n'
        'Always make sure to format the expression as a valid Python expression that can be evaluated by eval().\n'
        'Do not include natural language in the expression passed to calculate().\n'
        'If the user\'s request is unclear, ask for clarification.\n'
        'Available functions: abs, round, pow, sum, max, min, sqrt, sin, cos, tan\n'
        'Available constants: pi, e\n'
        'Set response status to input_required if the user needs to provide more information.\n'
        'Set response status to error if there is an error while processing the calculation.\n'
        'Set response status to completed if the calculation is complete.'
    )

    def __init__(self):
        # Use Ollama's llama2 model via LangChain Ollama integration
        logger.info("[Calculator Agent] Initializing with model acidtib/qwen2.5-coder-cline:7b")
        self.model = ChatOllama(model="acidtib/qwen2.5-coder-cline:7b", temperature=0)
        self.tools = [calculate]

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
            if (
                structured_response.status == 'input_required'
                or structured_response.status == 'error'
            ):
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        error_msg = 'We are unable to process your request at the moment. Please try again.'
        logger.error(f"[Calculator Agent] Error getting response: {error_msg}")
        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': error_msg,
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
