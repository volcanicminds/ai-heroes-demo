import logging
import traceback

from collections.abc import AsyncIterable
from typing import Any

from common.server import utils
from common.server.task_manager import InMemoryTaskManager
from common.types import (
    Artifact,
    InternalError,
    InvalidParamsError,
    JSONRPCResponse,
    Message,
    PushNotificationConfig,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    Task,
    TaskIdParams,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TextPart,
)

from agent import LangchainAgent # Import the LangchainAgent from the local module

logger = logging.getLogger(__name__)


class AgentTaskManager(InMemoryTaskManager):
    """Agent Task Manager, handles task routing and response packing."""

    def __init__(self, agent: LangchainAgent):
        super().__init__()
        self.agent = agent

    async def _stream_generator(
        self, request: SendTaskRequest
    ) -> AsyncIterable[SendTaskResponse]:
        raise NotImplementedError('Streaming is not implemented for this agent.')

    async def on_send_task(
        self, request: SendTaskRequest
    ) -> SendTaskResponse | AsyncIterable[SendTaskResponse]:
        logger.info(f"[A2A] Received 'tasks/send' request: id={request.id}, params={request.params}")
        # Validate input modes (assuming only text is supported for now)
        if not utils.are_modalities_compatible(
            request.params.acceptedOutputModes,
            LangchainAgent.SUPPORTED_CONTENT_TYPES,
        ):
            logger.warning(
                'Unsupported output mode. Received %s, Support %s',
                request.params.acceptedOutputModes,
                LangchainAgent.SUPPORTED_CONTENT_TYPES,
            )
            return utils.new_incompatible_types_error(request.id)

        task_send_params: TaskSendParams = request.params
        try:
            await self.upsert_task(task_send_params)
        except Exception as e:
            logger.error(f"[on_send_task] Error during upsert_task: {e}")
            logger.error(traceback.format_exc())
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message=f'Error storing task: {e}'
                ),
            )

        # Extract the user query
        try:
            query = self._get_user_query(task_send_params)
        except ValueError as e:
            logger.error(f"[on_send_task] Error extracting user query: {e}")
            return SendTaskResponse(
                id=request.id,
                error=InvalidParamsError(message=str(e)),
            )

        # Invoke the Langchain agent
        try:
            logger.info(f"[A2A] Invoking Langchain agent with query: {query}, sessionId={task_send_params.sessionId}")
            agent_response = self.agent.invoke(
                query, task_send_params.sessionId
            )
            logger.info(f"[A2A] Agent response for request id={request.id}: {agent_response}")
        except Exception as e:
            logger.error(f'Error invoking agent: {e}')
            # Update task status to failed
            await self.update_store(
                request.params.id, TaskStatus(state=TaskState.FAILED), None
            )
            return SendTaskResponse(
                id=request.id,
                error=InternalError(message=f'Error invoking agent: {e}'),
            )

        # Process and format the agent's response
        logger.info(f"[on_send_task] Returning processed agent response for request id={request.id}")
        return await self._process_agent_response(request, agent_response)


    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        logger.info(f"[A2A] Received 'tasks/sendSubscribe' request: id={request.id}, params={request.params}. Streaming is not implemented.")
        # Streaming is not implemented for this agent
        return JSONRPCResponse(
            id=request.id,
            error=InvalidParamsError(message='Streaming is not supported by this agent.'),
        )

    async def on_resubscribe_to_task(
        self, request
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        logger.info(f"[A2A] Received 'tasks/resubscribe' request: id={request.id}, params={request.params}. Resubscription is not applicable as streaming is not supported.")
        # Resubscription is not applicable as streaming is not supported
        return JSONRPCResponse(
            id=request.id,
            error=InvalidParamsError(message='Resubscription is not supported as streaming is not enabled.'),
        )

    async def _process_agent_response(
        self, request: SendTaskRequest, agent_response: dict
    ) -> SendTaskResponse:
        """Processes the agent's response and updates the task store."""
        logger.info(f"[_process_agent_response] Processing agent response for request id={request.id}: {agent_response}")
        task_send_params: TaskSendParams = request.params
        task_id = task_send_params.id
        history_length = task_send_params.historyLength
        task_status = None
        artifact = None

        parts = [{'type': 'text', 'text': agent_response.get('content', '')}]
        logger.info(f"[_process_agent_response] Extracted parts: {parts}")

        if agent_response.get('require_user_input', False):
            logger.info("[_process_agent_response] Agent response requires user input.")
            task_status = TaskStatus(
                state=TaskState.INPUT_REQUIRED,
                message=Message(role='agent', parts=parts),
            )
        elif agent_response.get('is_task_complete', False):
            logger.info("[_process_agent_response] Agent response indicates task is complete.")
            task_status = TaskStatus(state=TaskState.COMPLETED)
            artifact = Artifact(parts=parts)
        else:
             # Default to completed if no specific state is indicated
            logger.info("[_process_agent_response] Defaulting task status to COMPLETED.")
            task_status = TaskStatus(state=TaskState.COMPLETED)
            artifact = Artifact(parts=parts)

        logger.info(f"[_process_agent_response] Updating store for task {task_id} with status: {task_status.state}")
        task = await self.update_store(
            task_id, task_status, None if artifact is None else [artifact]
        )
        logger.info(f"[_process_agent_response] Store updated. Appending task history.")
        task_result = self.append_task_history(task, history_length)
        logger.info(f"[_process_agent_response] Task history appended. Returning SendTaskResponse.")

        return SendTaskResponse(id=request.id, result=task_result)

    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        logger.info(f"[_get_user_query] Extracting user query for task {task_send_params.id}.")
        if not task_send_params.message or not task_send_params.message.parts:
            logger.error("[_get_user_query] Message or message parts are missing.")
            raise ValueError("Message or message parts are missing in the request.")

        # Find the first text part
        for part in task_send_params.message.parts:
            if isinstance(part, TextPart):
                logger.info(f"[_get_user_query] Found text part: {part.text}")
                return part.text

        logger.error("[_get_user_query] No text parts found in the message.")
        raise ValueError('No text parts found in the message.')
