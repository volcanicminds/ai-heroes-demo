"""This file serves as the main entry point for the application.

It initializes the A2A server, defines the agent's capabilities,
and starts the server to handle incoming requests.
"""

import logging

import click

from agent import ContentGenerationCrew
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
    """Entry point for the A2A + CrewAI Content generation sample."""
    try:
        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id='content_generator',
            name='Content Generator',
            description=(
                'A collaborative crew of AI agents that work together to plan,'
                ' write, and edit high-quality content based on your prompt.'
            ),
            tags=['generate content', 'planning', 'writing', 'editing'],
            examples=[
                'Write an article about AI and its impact on society',
                'Create technical documentation for a new feature',
                'Generate a blog post about machine learning'
            ],
        )

        agent_card = AgentCard(
            name='Content Generation Crew',
            description=(
                'A specialized team of AI agents that collaborate to create'
                ' high-quality content through planning, writing, and editing.'
            ),
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=ContentGenerationCrew.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=ContentGenerationCrew.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=ContentGenerationCrew()),
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
