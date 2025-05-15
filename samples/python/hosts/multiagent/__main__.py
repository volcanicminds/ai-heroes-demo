"""
Prompt-based CLI for the multiagent host project.

- Discovers remote agents using A2A agent discovery
- Prompts the user for input
- Routes the prompt to the best agent
- Sends the prompt as a task to the selected agent
- Polls for the result and displays it
"""

import sys
import asyncio
from discovery import discover_agents
from router import route_prompt
from common.types import AgentCard, Message, TextPart, TaskSendParams, TaskState
from remote_agent_connection import RemoteAgentConnections
import uuid

AGENT_URLS = ["http://localhost:10000", "http://localhost:10001"]

# Helper to convert agent card dict to AgentCard object
def agent_card_from_dict(card_dict):
    # Remove extra fields not in AgentCard
    card_dict = dict(card_dict)
    card_dict.pop('base_url', None)
    return AgentCard(**card_dict)

async def main_async():
    print("Discovering agents...")
    agents = discover_agents(AGENT_URLS)
    if not agents:
        print("No agents discovered. Exiting.")
        sys.exit(1)
    print(f"Discovered {len(agents)} agent(s).\n")
    for idx, agent in enumerate(agents):
        print(f"[{idx}] {agent.get('name', 'Unknown')} @ {agent['base_url']}")
    # Build RemoteAgentConnections for each agent
    connections = {}
    for agent in agents:
        try:
            card = agent_card_from_dict(agent)
            connections[card.name] = RemoteAgentConnections(card)
        except Exception as e:
            print(f"Failed to create connection for agent {agent.get('name', 'Unknown')}: {e}")
    print("\nType ':q' or 'quit' to exit.")
    session_id = str(uuid.uuid4())
    while True:
        prompt = input("\nEnter your prompt: ").strip()
        if prompt in (":q", "quit"):
            print("Exiting.")
            break
        if not prompt:
            continue
        # Route the prompt to the best agent
        selected_agent = route_prompt(prompt, agents)
        if not selected_agent:
            print("No suitable agent found.")
            continue
        agent_name = selected_agent.get('name')
        print(f"Routing to agent: {agent_name or 'Unknown'} @ {selected_agent['base_url']}")
        connection = connections.get(agent_name)
        if not connection:
            print(f"No connection found for agent {agent_name}.")
            continue
        # Build the message and payload
        task_id = str(uuid.uuid4())
        message = Message(
            role='user',
            parts=[TextPart(text=prompt)],
            metadata=None
        )
        request = TaskSendParams(
            id=task_id,
            sessionId=session_id,
            message=message,
            acceptedOutputModes=['text'],
            metadata={'conversation_id': session_id}
        )
        print("Task sent. Waiting for result (task id: {} )...".format(task_id))
        # Streaming support if agent supports it
        try:
            print("DEBUG: Payload being sent:", request.model_dump(exclude_none=True))
            printed_during_streaming = False  # Flag to track output during streaming
            if connection.card.capabilities.streaming:
                # Use streaming if supported
                print("[INFO] Agent supports streaming. Streaming output:")
                async for result in connection.agent_client.send_task_streaming(request.model_dump(exclude_none=True)):
                    status = getattr(result.result, 'status', None)
                    message = getattr(status, 'message', None) if status else None
                    parts = getattr(message, 'parts', None) if message else None                    
                    if parts:
                        for part in parts:
                            if hasattr(part, 'text') and part.text:
                                print(part.text)
                                printed_during_streaming = True
                        continue
                    artifact = getattr(result.result, 'artifact', None)
                    artifact_parts = getattr(artifact, 'parts', None) if artifact else None
                    if artifact_parts:
                        for part in artifact_parts:
                            if hasattr(part, 'text') and part.text:
                                print(part.text)
                                printed_during_streaming = True
                        continue
                    # Print the full event for debugging if not minimal
                    print(f'stream event => {getattr(result, "model_dump_json", lambda **_: str(result))()}')
                # After streaming, fetch the final task result
                task_response = await connection.agent_client.get_task({'id': task_id})
                task = getattr(task_response, 'result', task_response)  # Unwrap if needed
            else:
                # Non-streaming fallback
                task = await connection.send_task(request, None)
        except Exception as e:
            print(f"Failed to send or receive task: {e}")
            continue

        # Only print final result if nothing was printed during streaming or if not streaming
        if not printed_during_streaming:
            artifacts = getattr(task, 'artifacts', []) or []
            if artifacts:
                print("\nArtifacts:")
                for art in artifacts:
                    # Print the first text part if available
                    parts = getattr(art, 'parts', [])
                    text = None
                    for part in parts:
                        if getattr(part, 'type', None) == 'text':
                            text = getattr(part, 'text', str(part))
                            break
                    print(f"- {getattr(art, 'name', 'artifact')}: {text}")
            else:
                # Fallback: print the last message
                status = getattr(task, 'status', None)
                if status and status.message and status.message.parts:
                    last_part = status.message.parts[-1]
                    if getattr(last_part, 'type', None) == 'text':
                        print("\nResponse:")
                        print(getattr(last_part, 'text', str(last_part)))
                    else:
                        print("No text output from agent (final state).")
                else:
                    print("No output from agent (final state).")
                    # Debug: print the full task object for inspection
                    print("Raw task result (final state):", task)

# Entry point
if __name__ == "__main__":
    asyncio.run(main_async()) 