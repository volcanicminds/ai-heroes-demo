"""This file serves as the main entry point for the application.

It initializes the A2A server, defines the agent's capabilities,
and starts the server to handle incoming requests.
"""

import logging
import os

import click

from agent import TextGenerationAgent
from common.server import A2AServer
from common.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MissingAPIKeyError,
)
from dotenv import load_dotenv
from task_manager import AgentTaskManager


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10001)
def main(host, port):
    """Entry point for the A2A + CrewAI Text generation sample."""
    try:
        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id='text_generator',
            name='Text Generator',
            description=(
                'It generates text based on the prompt.'
            ),
            tags=['generate text'],
            examples=['Generate a text based on the prompt'],
        )

        agent_card = AgentCard(
            name='Text Generator Agent',
            description=(
                'It generates text based on the prompt.'
            ),
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=TextGenerationAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=TextGenerationAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=TextGenerationAgent()),
            host=host,
            port=port,
        )
        logger.info(f'Starting server on {host}:{port}')
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()
