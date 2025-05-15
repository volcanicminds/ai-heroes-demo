import requests
from typing import List, Dict, Any

# This function fetches the agent card from the well-known URL for each base URL provided.
def discover_agents(agent_base_urls: List[str]) -> List[Dict[str, Any]]:
    """
    Discover A2A agents by fetching their agent cards from the well-known path.
    Returns a list of agent card dicts with their base URL included.
    """
    discovered = []
    for base_url in agent_base_urls:
        try:
            url = f"{base_url}/.well-known/agent.json"
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            agent_card = response.json()
            agent_card['base_url'] = base_url  # Add base_url for later reference
            discovered.append(agent_card)
        except Exception as e:
            # Log or print error, but continue
            print(f"Failed to fetch agent card from {base_url}: {e}")
    return discovered

# Example usage (for testing):
if __name__ == "__main__":
    agents = discover_agents(["http://localhost:10000", "http://localhost:10001"])
    # Log all discovered agent cards
    print(f"Discovered agent cards: {agents}")  # This prints the list of all agent cards
    for agent in agents:
        print(agent) 