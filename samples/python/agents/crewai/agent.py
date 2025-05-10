"""Crew AI based sample for A2A protocol.

Handles the agents and also presents the tools required.
"""

import logging

from collections.abc import AsyncIterable
from typing import Any

from common.utils.in_memory_cache import InMemoryCache
from crewai import LLM, Agent, Crew, Task
from crewai.process import Process
from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()

logger = logging.getLogger(__name__)

class TextGenerationAgent:
    """Agent that generates text based on user prompts."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        self.model = LLM(
            model="ollama/llama3.2:latest",
            base_url="http://localhost:11434"
        )
        self.text_creator_agent = Agent(
            role='Text Creation Expert',
            goal=(
                "Generate an text based on the user's text prompt."
            ),
            backstory=(
                'You are a text creation expert powered by AI. You specialize in taking'
                ' the user prompt and creating text based on the prompt.'
            ),
            verbose=False,
            allow_delegation=False,
            llm=self.model,
        )

        self.text_creation_task = Task(
            description=(
                "Receive a user prompt: '{user_prompt}'.\nAnalyze the prompt and"
                ' identify if you need to create a new text or edit an existing'
                ' one. Look for pronouns like this, that etc in the prompt, they'
                ' might provide context, rewrite the prompt to include the'
                ' context.If creating a new text, ignore any text provided as'
                " input context.Use the 'Text Generator' tool to for your text"
                ' creation or modification. The tool will expect a prompt which is'
                ' the {user_prompt} and the session_id which is {session_id}.'
            ),
            expected_output='The id of the generated text',
            agent=self.text_creator_agent,
        )

        self.text_crew = Crew(
            agents=[self.text_creator_agent],
            tasks=[self.text_creation_task],
            process=Process.sequential,
            verbose=False,
        )

    def invoke(self, query, session_id) -> str:
        """Kickoff CrewAI and return the response."""

        inputs = {
            'user_prompt': query,
            'session_id': session_id,
        }
        logger.info(f'Inputs {inputs}')
        print(f'Inputs {inputs}')
        response = self.text_crew.kickoff(inputs)
        return response

    async def stream(self, query: str) -> AsyncIterable[dict[str, Any]]:
        """Streaming is not supported by CrewAI."""
        raise NotImplementedError('Streaming is not supported by CrewAI.')
