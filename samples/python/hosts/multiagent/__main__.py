"""
Prompt-based CLI for the multiagent host project.

- Discovers remote agents using A2A agent discovery
- Prompts the user for input
- Routes the prompt to the best agent
- Sends the prompt as a task to the selected agent
- Polls for the result and displays it
"""

import sys
from discovery import discover_agents
from router import route_prompt
from client import send_task, poll_task

AGENT_URLS = ["http://localhost:10000", "http://localhost:10001"]

def main():
    print("Discovering agents...")
    agents = discover_agents(AGENT_URLS)
    if not agents:
        print("No agents discovered. Exiting.")
        sys.exit(1)
    print(f"Discovered {len(agents)} agent(s).\n")
    for idx, agent in enumerate(agents):
        print(f"[{idx}] {agent.get('name', 'Unknown')} @ {agent['base_url']}")
    print("\nType ':q' or 'quit' to exit.")
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
        print(f"Routing to agent: {selected_agent.get('name', 'Unknown')} @ {selected_agent['base_url']}")
        # Send the task
        task_id = send_task(selected_agent['base_url'], prompt)
        if not task_id:
            print("Failed to send task.")
            continue
        print(f"Task sent. Waiting for result (task id: {task_id})...")
        # Poll for result
        result = poll_task(selected_agent['base_url'], task_id)
        if not result:
            print("No result received or task failed.")
            continue
        # Print the result (look for artifacts or messages)
        artifacts = result.get("artifacts", [])
        if artifacts:
            print("\nArtifacts:")
            for art in artifacts:
                print(f"- {art.get('name', 'artifact')}: {art.get('parts', [{}])[0].get('text', str(art.get('parts', [{}])[0]))}")
        else:
            # Fallback: print the last message
            messages = result.get("messages", [])
            if messages:
                print("\nResponse:")
                print(messages[-1].get("parts", [{}])[0].get("text", str(messages[-1].get("parts", [{}])[0])))
            else:
                print("No output from agent.")

if __name__ == "__main__":
    main() 