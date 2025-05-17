"""Agent Task Manager."""

import logging

from collections.abc import AsyncIterable

from agent import ContentGenerationCrew
from common.server import utils
from common.server.task_manager import InMemoryTaskManager
from common.types import (
    Message,
    Task,
    JSONRPCResponse,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TextPart,
)


logger = logging.getLogger(__name__)


class AgentTaskManager(InMemoryTaskManager):
    """Agent Task Manager, handles task routing and response packing."""

    def __init__(self, agent: ContentGenerationCrew):
        super().__init__()
        self.agent = agent

    async def _stream_generator(
        self, request: SendTaskRequest
    ) -> AsyncIterable[SendTaskResponse]:
        raise NotImplementedError('Not implemented')

    async def on_send_task(
        self, request: SendTaskRequest
    ) -> SendTaskResponse | AsyncIterable[SendTaskResponse]:
        ## only support text output at the moment
        if not utils.are_modalities_compatible(
            request.params.acceptedOutputModes,
            ContentGenerationCrew.SUPPORTED_CONTENT_TYPES,
        ):
            logger.warning(
                'Unsupported output mode. Received %s, Support %s',
                request.params.acceptedOutputModes,
                ContentGenerationCrew.SUPPORTED_CONTENT_TYPES,
            )
            return utils.new_incompatible_types_error(request.id)

        task_send_params: TaskSendParams = request.params
        await self.upsert_task(task_send_params)

        return await self._invoke(request)

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        error = self._validate_request(request)
        if error:
            return error

        await self.upsert_task(request.params)


    async def _invoke(self, request: SendTaskRequest) -> SendTaskResponse:
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        try:
            result = self.agent.invoke(query, task_send_params.sessionId)
        except Exception as e:
            logger.error('Error invoking agent: %s', e)
            raise ValueError(f'Error invoking agent: {e}') from e
        
        try:
            print(f'Final Result ===> {result}')

            agent_message = Message(
                role='agent',
                parts=[TextPart(text=str(result))],
                metadata=None
            )
            print(f'Agent message ===> {agent_message}')

            status = TaskStatus(
                state=TaskState.COMPLETED,
                message=agent_message
            )
            print(f'Task status ===> {status}')

            task = Task(
                id=task_send_params.id,
                sessionId=task_send_params.sessionId,
                status=status,
                artifacts=None,
                history=None,
                metadata=None
            )

            print(f'Task ===> {task}')
        
        except Exception as e:
            logger.error('Error invoking agent: %s', e)
            raise ValueError(f'Error invoking agent: {e}') from e
        
        return SendTaskResponse(id=request.id, result=task)

    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        part = task_send_params.message.parts[0]
        if not isinstance(part, TextPart):
            raise ValueError('Only text parts are supported')

        return part.text
