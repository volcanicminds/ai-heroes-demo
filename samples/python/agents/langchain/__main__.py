"""This file serves as the main entry point for the application.

It initializes the A2A server, defines the agent's capabilities,
and starts the server to handle incoming requests.
"""

import logging

import click

from agent import LangchainAgent
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
@click.option('--port', 'port', default=10002) # Use a different default port
def main(host, port):
    """Starts the Langchain Agent server."""
    try:
        capabilities = AgentCapabilities(streaming=False, pushNotifications=False) # Start without streaming and push notifications
        skill = AgentSkill(
            id='langchain_agent',
            name='Langchain Agent',
            description='An agent that can discover and route messages to other agents.',
            tags=['langchain', 'multi-agent', 'routing'],
            examples=['Discover agents', 'Send this message to the text generator agent: generate a short story about a dog'],
        )
        agent_card = AgentCard(
            name='Langchain Router Agent',
            description='An agent that can discover and route messages to other agents.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=LangchainAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=LangchainAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        logger.info("AgentCard created.")

        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=LangchainAgent(),
            ),
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
