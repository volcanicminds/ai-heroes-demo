# LangGraph Currency Agent

This project demonstrates a currency conversion agent built using LangGraph and the A2A protocol. The agent supports multi-turn conversations, real-time streaming, and integration with external APIs for currency exchange rates.

## Features

- **Mathematical Calculations**: The agent can evaluate mathematical expressions using a secure and restricted environment.
- **Multi-turn Conversations**: Maintains context across interactions to handle complex queries.
- **Real-time Streaming**: Provides incremental updates during processing.
- **A2A Protocol Integration**: Enables standardized communication with other agents.
- **Extensibility**: Easily integrates additional tools and APIs.

## Technical Overview

### Core Components

1. **LangGraph Framework**: Utilizes LangGraph's ReAct agent pattern for reasoning and tool usage.
2. **Mathematical Calculation Tool**: A custom tool for evaluating mathematical expressions securely.
3. **Response Formatting**: Ensures consistent and numeric-only responses.
4. **Streaming Support**: Allows real-time updates during task execution.
5. **A2A Protocol**: Facilitates communication between agents and clients.

### Key Classes and Functions

- `calculate(expression: str) -> dict`: Evaluates mathematical expressions securely.
- `ResponseFormat`: Formats and validates responses to ensure numeric-only outputs.
- `CalculationAgent`: Manages the agent's lifecycle, including query processing and streaming.

## Prerequisites

- Python 3.13 or higher
- Required Python packages (install via `pip install -r requirements.txt`)
- Access to an LLM and API key

## Setup and Usage

1. Clone the repository and navigate to the project directory:

   ```bash
   cd app/agents/langgraph
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create an environment file with your API key:

   ```bash
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

4. Run the agent:

   ```bash
   uv run .
   ```

5. Use an A2A client to interact with the agent:

   ```bash
   cd ../hosts/cli
   uv run .
   ```

## Example Usage

### Single-turn Query

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tasks/send",
  "params": {
    "id": "123",
    "sessionId": "abc123",
    "message": {
      "role": "user",
      "parts": [{ "type": "text", "text": "What is 5 plus 3?" }]
    }
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "status": "completed",
    "message": "8"
  }
}
```

### Multi-turn Query

1. User asks for a calculation:

   ```json
   {
     "message": "What is the square root of 16?"
   }
   ```

2. Agent responds with the result:

   ```json
   {
     "message": "4"
   }
   ```

## Limitations

- Limited to text-based input and output.
- Session-based memory; does not persist across server restarts.
- Relies on external APIs for currency exchange rates.

## Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [A2A Protocol Documentation](https://google.github.io/A2A/#/documentation)
- [Frankfurter API](https://www.frankfurter.app/docs/)
