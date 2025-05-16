# CLI Host Application

The CLI host application demonstrates the capabilities of an A2AClient by enabling text-based collaboration with a remote agent. It connects to an A2A server, retrieves the server's AgentCard, and facilitates interactive communication with the agent.

## Features

- **AgentCard Retrieval**: Fetches and displays the AgentCard of the connected A2A server.
- **Interactive Prompting**: Accepts user input via the terminal and sends it to the agent.
- **File Attachments**: Allows users to attach files to their messages.
- **Streaming Support**: Displays streaming responses if the server supports it.
- **Push Notifications**: Optionally supports push notifications for task updates.
- **Session Management**: Maintains session context for ongoing interactions.
- **Task History**: Optionally retrieves and displays task history.

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/)
- A running A2A server (e.g., LangChain Router Agent)

## Running the CLI

1. Navigate to the CLI sample directory:
   ```bash
   cd samples/python/hosts/cli
   ```
2. Start the CLI client:
   ```bash
   uv run . --agent [url-of-your-a2a-server]
   ```
   Replace `[url-of-your-a2a-server]` with the URL of the A2A server you want to connect to (e.g., `http://localhost:10002`).

## Command-Line Options

- `--agent`: The URL of the A2A server to connect to (default: `http://localhost:10002`).
- `--session`: Specify a session ID to continue an existing session (default: `0` for a new session).
- `--history`: Enable task history retrieval (default: `False`).
- `--use_push_notifications`: Enable push notifications for task updates (default: `False`).
- `--push_notification_receiver`: Specify the URL for receiving push notifications (default: `http://localhost:5000`).

## Example Usage

### Start a New Session

```bash
uv run . --agent http://localhost:10002
```

### Continue an Existing Session

```bash
uv run . --agent http://localhost:10002 --session [session-id]
```

### Enable Push Notifications

```bash
uv run . --agent http://localhost:10002 --use_push_notifications
```

### Retrieve Task History

```bash
uv run . --agent http://localhost:10002 --history
```

## Notes

- The CLI client is designed to work with any A2A-compliant server.
- Push notifications require the server to support and configure a `.well-known/jwks.json` endpoint.
- This is a sample application and is not intended for production use.
