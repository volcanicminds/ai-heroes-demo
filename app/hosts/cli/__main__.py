import asyncio
import urllib

from uuid import uuid4

import asyncclick as click

from common.client import A2ACardResolver, A2AClient
from common.types import TaskState
from common.utils.push_notification_auth import PushNotificationReceiverAuth


@click.command()
@click.option('--agent', default='http://localhost:10002')
@click.option('--session', default=0)
@click.option('--history', default=False)
@click.option('--use_push_notifications', default=False)
@click.option('--push_notification_receiver', default='http://localhost:5000')
async def cli(
    agent,
    session,
    history,
    use_push_notifications: bool,
    push_notification_receiver: str,
):
    card_resolver = A2ACardResolver(agent)
    card = card_resolver.get_agent_card()

    print('\n🤖 Successfully connected to AI Agent')
    print('📋 Available capabilities:')
    print(card.model_dump_json(exclude_none=True, indent=2))

    notif_receiver_parsed = urllib.parse.urlparse(push_notification_receiver)
    notification_receiver_host = notif_receiver_parsed.hostname
    notification_receiver_port = notif_receiver_parsed.port

    if use_push_notifications:
        from hosts.cli.push_notification_listener import (
            PushNotificationListener,
        )

        notification_receiver_auth = PushNotificationReceiverAuth()
        await notification_receiver_auth.load_jwks(
            f'{agent}/.well-known/jwks.json'
        )

        push_notification_listener = PushNotificationListener(
            host=notification_receiver_host,
            port=notification_receiver_port,
            notification_receiver_auth=notification_receiver_auth,
        )
        push_notification_listener.start()

    client = A2AClient(agent_card=card, timeout=300.0)  # 5 minutes timeout
    if session == 0:
        sessionId = uuid4().hex
    else:
        sessionId = session

    continue_loop = True
    streaming = card.capabilities.streaming

    while continue_loop:
        taskId = uuid4().hex
        print('\n🔄 Starting a new conversation')
        continue_loop = await completeTask(
            client,
            streaming,
            use_push_notifications,
            notification_receiver_host,
            notification_receiver_port,
            taskId,
            sessionId,
        )

        if history and continue_loop:
            print('\n📜 Conversation History:')
            task_response = await client.get_task(
                {'id': taskId, 'historyLength': 10}
            )
            print(
                task_response.model_dump_json(
                    include={'result': {'history': True}}
                )
            )


def to_dict(obj):
    """Convert an object to a dict if possible."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'model_dump') and callable(obj.model_dump):
        return obj.model_dump()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return {}


def print_parts(parts):
    if parts:
        for part in parts:
            print(part.text)
        return True
    return False


async def completeTask(
    client: A2AClient,
    streaming,
    use_push_notifications: bool,
    notification_receiver_host: str,
    notification_receiver_port: int,
    taskId,
    sessionId,
):
    prompt = click.prompt(
        '\nWhat do you want to send to the agent? (:q or quit to exit)'
    )
    if prompt == ':q' or prompt == 'quit':
        return False

    message = {
        'role': 'user',
        'parts': [
            {
                'type': 'text',
                'text': prompt,
            }
        ],
    }

    payload = {
        'id': taskId,
        'sessionId': sessionId,
        'acceptedOutputModes': ['text'],
        'message': message,
    }

    if use_push_notifications:
        payload['pushNotification'] = {
            'url': f'http://{notification_receiver_host}:{notification_receiver_port}/notify',
            'authentication': {
                'schemes': ['bearer'],
            },
        }

    taskResult = None
    if streaming:
        response_stream = client.send_task_streaming(payload)
        async for result in response_stream:
            status = getattr(result.result, 'status', None)
            message = getattr(status, 'message', None) if status else None
            parts = getattr(message, 'parts', None) if message else None
            if parts:
                for part in parts: print(part.text)
                continue
            artifact = getattr(result.result, 'artifact', None)
            artifact_parts = getattr(artifact, 'parts', None) if artifact else None
            if artifact_parts:
                for part in artifact_parts: print(part.text)
                continue
            result_dict = to_dict(result.result)
            status_dict = to_dict(result_dict.get('status', {}))
            # Remove keys with value None for minimality check
            minimal_result_keys = {k for k, v in result_dict.items() if v is not None}
            # Allow 'message' in status_dict, as it may be present and None
            if minimal_result_keys.issubset({'id', 'status', 'final'}) \
                and set(status_dict.keys()).issubset({'state', 'timestamp', 'message'}):
                continue
            print(f'\n🔄 Stream update: {result.model_dump_json(exclude_none=True, indent=2)}')

        taskResult = await client.get_task({'id': taskId})
    else:
        # Display thinking animation
        print('\n🤔 AI Agent is thinking...')
        taskResult = await client.send_task(payload)
        # Clear thinking message
        print('\r' + ' ' * 30, end='')
        
        try:
            # First try to get text from status message
            if hasattr(taskResult.result.status, 'message') and taskResult.result.status.message:
                full_text = ""
                for part in taskResult.result.status.message.parts:
                    if hasattr(part, 'text'):
                        full_text += part.text
                if full_text:
                    print(full_text)
                    return True

            # Then try to get text from artifacts
            if hasattr(taskResult.result, 'artifacts') and taskResult.result.artifacts:
                for artifact in taskResult.result.artifacts:
                    if hasattr(artifact, 'parts'):
                        for part in artifact.parts:
                            if hasattr(part, 'text'):
                                print(part.text)
                return True

            # If no text content found, fall back to printing full JSON
            print(f'\n{taskResult.model_dump_json(exclude_none=True)}')
        except Exception as e:
            print(f'\n{taskResult.model_dump_json(exclude_none=True)}')

    ## if the result is that more input is required, loop again.
    state = TaskState(taskResult.result.status.state)
    if state.name == TaskState.INPUT_REQUIRED.name:
        return await completeTask(
            client,
            streaming,
            use_push_notifications,
            notification_receiver_host,
            notification_receiver_port,
            taskId,
            sessionId,
        )
    ## task is complete
    return True


if __name__ == '__main__':
    asyncio.run(cli())
