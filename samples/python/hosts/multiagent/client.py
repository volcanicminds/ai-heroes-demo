import requests
import time
from typing import Dict, Any, Optional

# Send a task to a remote agent using the A2A protocol
# Returns the task id

def send_task(agent_base_url: str, message: str) -> Optional[str]:
    """
    Sends a task to the remote agent. Returns the task id if successful, else None.
    """
    url = f"{agent_base_url}/tasks"
    payload = {
        "message": {
            "role": "user",
            "parts": [
                {"type": "text", "text": message}
            ]
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("id")
    except Exception as e:
        print(f"Failed to send task to {agent_base_url}: {e}")
        return None

# Poll for task completion
# Returns the completed task object or None

def poll_task(agent_base_url: str, task_id: str, poll_interval: float = 1.0, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    """
    Polls the remote agent for task completion. Returns the completed task object or None if timeout.
    """
    url = f"{agent_base_url}/tasks/{task_id}"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            state = data.get("state")
            if state == "completed":
                return data
            elif state in ("failed", "canceled"):
                print(f"Task {task_id} failed or was canceled.")
                return None
        except Exception as e:
            print(f"Error polling task {task_id} from {agent_base_url}: {e}")
            return None
        time.sleep(poll_interval)
    print(f"Timeout waiting for task {task_id} to complete.")
    return None 