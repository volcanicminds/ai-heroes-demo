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
import uuid

from google.adk import Runner

from host_agent import HostAgent

AGENT_URLS = ["http://localhost:10000", "http://localhost:10001"]

async def main_async():
    print("Initializing HostAgent...")
    
    def on_sub_agent_task_update(task_update_info):
        pass

    host_agent = HostAgent(
        remote_agent_addresses=AGENT_URLS,
        task_callback=on_sub_agent_task_update
    )

    session_id = str(uuid.uuid4())

    print("HostAgent is active and has discovered the following remote agents:")
    if host_agent.cards:
        for name, card in host_agent.cards.items():
            print(f"  - Name: {name}, Description: {card.description}")
    else:
        print("  No remote agents discovered by HostAgent.")
    print("-" * 30)
    print('Type ":q" or "quit" to exit.')

    agent = host_agent.create_agent()
    host_runner = Runner(
            app_name=self.app_name,
            agent=agent,
            artifact_service=self._artifact_service,
            session_service=self._session_service,
            memory_service=self._memory_service,
        )

    while True:
        prompt = input("\nEnter your prompt for the HostAgent: ").strip()
        if prompt.lower() in (":q", "quit"):
            print("Exiting.")
            break
        if not prompt:
            continue

        print("\nHostAgent processing...")
        response_from_host = await agent.run(prompt, session_id)
        
        print(f"\nResponse:\n{response_from_host}")
        print("-" * 30)

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
