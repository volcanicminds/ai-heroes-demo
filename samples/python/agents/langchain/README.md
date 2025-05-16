# Langchain Router Agent with A2A Protocol

This project implements a Langchain-based router agent that leverages the A2A protocol to discover and route messages to other agents. It is designed to facilitate multi-agent communication by dynamically forwarding tasks to the most suitable agent based on their capabilities.

## Overview

The Langchain Router Agent acts as a middleware for routing tasks between A2A agents. It uses Langchain's tools and models to process incoming messages, discover other agents, and forward tasks to the appropriate agent. The agent is fully compliant with the A2A protocol, ensuring standardized communication.

### Key Features

- **Agent Discovery**: Automatically identifies other A2A agents using predefined URLs.
- **Dynamic Message Routing**: Routes tasks to the most suitable agent based on their capabilities.
- **A2A Protocol Integration**: Ensures seamless communication with other agents.
- **Extensible Design**: Built with Langchain, allowing easy customization and extension.

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) for environment management
- Access to a Langchain-supported Language Model (e.g., Google Gemini, Ollama)

## Setup Instructions

1. **Navigate to the Project Directory**:

   ```bash
   cd samples/python/agents/langchain
   ```

2. **Set Up Environment Variables** (if required by your LLM):

   ```bash
   echo "YOUR_API_KEY_NAME=your_api_key_here" > .env
   ```

   Replace `YOUR_API_KEY_NAME` with the actual environment variable name (e.g., `GOOGLE_API_KEY`).

3. **Install Dependencies**:

   ```bash
   uv python pin 3.12
   uv venv
   source .venv/bin/activate
   uv install -r requirements.txt
   ```

4. **Configure the Language Model**:
   Open `agent.py` and replace the `model` initialization with your desired LLM configuration.

5. **Run the Agent**:

   ```bash
   uv run . --host 0.0.0.0 --port 8080
   ```

6. **Test the Agent**:
   Use an A2A client to send tasks to the agent:
   ```bash
   cd samples/python/hosts/cli
   uv run . --agent http://localhost:8080
   ```

## Technical Details

### Agent Workflow

1. **Message Processing**: The agent processes incoming messages using Langchain's tools.
2. **Agent Discovery**: It identifies other agents using the `discover_agents` tool.
3. **Message Routing**: The `route_message` tool forwards tasks to the selected agent and returns the response.

### Tools and Models

- **Langchain Tools**: `discover_agents` and `route_message` for agent discovery and task routing.
- **Language Model**: Configurable LLM for processing messages.
- **A2A Protocol**: Ensures standardized communication between agents.

## Limitations

- **Simplified Routing Logic**: The current implementation uses basic logic for task routing.
- **No Streaming Support**: Real-time streaming of responses is not yet implemented.
- **Predefined Agent URLs**: Discovery is limited to a static list of agent URLs.

## Future Improvements

- Enhance routing logic to support more complex decision-making.
- Add support for real-time streaming of responses.
- Implement dynamic agent discovery mechanisms.

## Learn More

- [Langchain Documentation](https://python.langchain.com/v0.2/docs/introduction/)
- [A2A Protocol Documentation](https://google.github.io/A2A/#/documentation)
