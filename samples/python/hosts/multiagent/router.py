from typing import List, Dict, Any

# Simple keyword-based router
# For now, checks if any keyword in the agent's card is in the prompt
# Otherwise, defaults to the first agent

def route_prompt(prompt: str, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Selects the best agent for the prompt based on keyword matching.
    Returns the selected agent's card.
    """
    prompt_lower = prompt.lower()
    for agent in agents:
        # Check name and description fields for keywords
        name = agent.get("name", "").lower()
        description = agent.get("description", "").lower()
        if name and name in prompt_lower:
            return agent
        if description and any(word in prompt_lower for word in description.split()):
            return agent
    # Default: return the first agent
    return agents[0] if agents else None 