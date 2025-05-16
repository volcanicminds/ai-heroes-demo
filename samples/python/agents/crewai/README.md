# CrewAI Agent with A2A Protocol

This project demonstrates a content generation agent built using [CrewAI](https://www.crewai.com/open-source) and exposed via the A2A protocol. The agent collaborates to generate high-quality content through planning, writing, and editing.

## Overview

The agent leverages CrewAI's capabilities to:

- Plan content structure based on user prompts.
- Write detailed and engaging content.
- Edit and refine the content for clarity and coherence.

The A2A protocol standardizes interactions, enabling seamless communication between clients and the agent.

## Key Features

- **Collaborative Content Generation**: A team of specialized agents (Planner, Writer, Editor) work together to produce high-quality content.
- **Sequential Workflow**: Tasks are executed in a structured sequence: Planning -> Writing -> Editing.
- **Logging and Debugging**: Detailed logs are generated for each step, aiding in debugging and analysis.
- **Customizable**: Easily extendable to support additional tasks or workflows.

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- [CrewAI](https://www.crewai.com/open-source) library
- Google API Key (for Gemini access)

## Setup Instructions

1. **Navigate to the project directory:**

   ```bash
   cd samples/python/agents/crewai
   ```

2. **Set up environment variables:**
   Create a `.env` file with your API key:

   ```bash
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

3. **Set up the Python environment:**

   ```bash
   uv python pin 3.12
   uv venv
   source .venv/bin/activate
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the agent:**

   ```bash
   uv run . --host 0.0.0.0 --port 10001
   ```

6. **Run the A2A client:**
   In a separate terminal:
   ```bash
   cd samples/python/hosts/cli
   uv run . --agent http://localhost:10001
   ```

## How It Works

1. **Planning**: The Planner agent creates a structured outline based on the user prompt.
2. **Writing**: The Writer agent generates detailed content from the outline.
3. **Editing**: The Editor agent refines the content for quality and coherence.

The workflow is managed by the `ContentGenerationCrew` class, which orchestrates the agents and tasks.

## File Structure

- `__main__.py`: Entry point for the application.
- `agent.py`: Defines the `ContentGenerationCrew` and its agents.
- `task_manager.py`: Manages task execution.
- `crew_execution.json`: Logs detailed execution steps.
- `outline.txt`, `draft.txt`, `final.txt`: Intermediate and final outputs of the content generation process.

## Limitations

- No support for streaming responses.
- Limited to single-turn interactions.

## Learn More

- [CrewAI Documentation](https://docs.crewai.com/introduction)
- [A2A Protocol Documentation](https://google.github.io/A2A/#/documentation)
- [Google Gemini API](https://ai.google.dev/gemini-api)
