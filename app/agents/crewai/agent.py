"""Crew AI based sample for A2A protocol.

Handles the agents and also presents the tools required.
"""

from collections.abc import AsyncIterable
from typing import Any
from crewai import LLM, Agent, Crew, Task
from crewai import Process
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv()

# Configure logging for better demo presentation
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Clean format for demo purposes
)

class ContentGenerationCrew:
    """Crew that generates content using a team of specialized agents."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        logger.info("\n🚀 Initializing AI Content Generation Crew...")
        self.model = LLM(
            model="ollama/llama3.2:latest",
            base_url="http://localhost:11434"
        )
        
        # Define the agents with enhanced demo-friendly logging
        def log_agent_action(agent_name: str, input_text: str, output_text: str):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n{'='*80}")
            print(f"⏰ {timestamp} | 🤖 {agent_name} is working...")
            print(f"{'='*80}")
            print(f"📥 INPUT:")
            print(f"{'-'*80}")
            print(f"{input_text}")
            print(f"\n📤 OUTPUT:")
            print(f"{'-'*80}")
            print(f"{output_text}")
            print(f"{'='*80}\n")

        # Define the agents
        self.planner = Agent(
            role="Content Planner",
            goal="Develop a comprehensive and structured content outline based on the user's prompt",
            backstory="An expert content strategist skilled at breaking down complex topics into manageable parts.",
            verbose=True,  # Enable verbose mode
            allow_delegation=False,
            llm=self.model,
            max_iterations=1  # Limit iterations for debugging
        )
        
        self.writer = Agent(
            role="Content Writer",
            goal="Produce captivating and informative content based on the outline",
            backstory="A versatile writer passionate about simplifying complex ideas.",
            verbose=True,  # Enable verbose mode
            allow_delegation=False,
            llm=self.model,
            max_iterations=1  # Limit iterations for debugging
        )
        
        self.editor = Agent(
            role="Content Editor",
            goal="Refine the content, ensuring clarity, coherence, and grammatical accuracy",
            backstory="A meticulous editor with a strong eye for detail.",
            verbose=True,  # Enable verbose mode
            allow_delegation=False,
            llm=self.model,
            max_iterations=1  # Limit iterations for debugging
        )

        # Define the tasks with enhanced logging
        self.planning_task = Task(
            description=(
                "Create a detailed content outline for the following prompt: '{user_prompt}'.\n"
                "Include main sections and key points that need to be addressed."
            ),
            expected_output="A structured outline with headings and bullet points",
            agent=self.planner,
            output_file="outline.txt"  # Add output file for logging
        )

        self.writing_task = Task(
            description=(
                "Transform the outline into a comprehensive response. Use clear language and"
                " examples where appropriate.\nThe user prompt is: '{user_prompt}'"
            ),
            expected_output="A comprehensive response with clear language and examples",
            agent=self.writer,
            output_file="draft.txt"  # Add output file for logging
        )

        self.editing_task = Task(
            description=(
                "Review and polish the content, ensuring quality and alignment with both the"
                " original prompt and outline.\nThe user prompt is: '{user_prompt}'"
            ),
            expected_output="A polished, error-free response with enhanced structure and tone",
            agent=self.editor,
            output_file="final.txt"  # Add output file for logging
        )

        # Assemble the crew with detailed logging
        self.content_crew = Crew(
            agents=[self.planner, self.writer, self.editor],
            tasks=[self.planning_task, self.writing_task, self.editing_task],
            process=Process.sequential,
            verbose=True,  # Enable crew-level verbose mode
            output_log_file="crew_execution.json"  # Save detailed execution logs in JSON format
        )

    def invoke(self, query, session_id) -> str:
        """Process content generation request."""
        logger.info(f"\n📋 New Content Generation Task")
        logger.info(f"🆔 Session: {session_id}")
        logger.info(f"❓ Query: {query}\n")
        
        inputs = {
            'user_prompt': query,
            'session_id': session_id,
        }
        
        logger.info("🎬 Starting CrewAI workflow...")
        
        try:
            response = self.content_crew.kickoff(inputs)
            logger.info("✅ Content generation completed successfully!")
            return response
        except Exception as e:
            logger.error(f"❌ Error during content generation: {str(e)}")
            raise

    async def stream(self, query: str) -> AsyncIterable[dict[str, Any]]:
        """Streaming is not supported by CrewAI."""
        raise NotImplementedError('Streaming is not supported by CrewAI.')
