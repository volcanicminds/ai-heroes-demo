import base64
import json
import uuid
import traceback

from common.client import A2ACardResolver
from common.types import (
    AgentCard,
    DataPart,
    Message,
    Part,
    Task,
    TaskSendParams,
    TaskState,
    TextPart,
)
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from remote_agent_connection import RemoteAgentConnections, TaskUpdateCallback


class HostAgent:
    """The host agent.

    This is the agent responsible for choosing which remote agents to send
    tasks to and coordinate their work.
    """

    def __init__(
        self,
        remote_agent_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        for address in remote_agent_addresses:
            card_resolver = A2ACardResolver(address)
            card = card_resolver.get_agent_card()
            remote_connection = RemoteAgentConnections(card)
            self.remote_agent_connections[card.name] = remote_connection
            self.cards[card.name] = card
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = '\n'.join(agent_info)

    def register_agent_card(self, card: AgentCard):
        remote_connection = RemoteAgentConnections(card)
        self.remote_agent_connections[card.name] = remote_connection
        self.cards[card.name] = card
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = '\n'.join(agent_info)

    def create_agent(self) -> Agent:
        return Agent(
            model="ollama/llama3.2:latest",
            base_url="http://localhost:11434",
            name='host_agent',
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                'This agent analyzes user requests and delegates them to the most '
                'appropriate specialized remote agent using its tools.'
            ),
            tools=[
                self.list_remote_agents,
                self.send_task,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        agent_descriptions = []
        if self.cards:
            for i, (name, card) in enumerate(self.cards.items()):
                description = card.description if card.description else 'No description available.'
                skills_info = ""
                if card.skills:
                    skills_list = []
                    for skill in card.skills:
                        skill_desc = skill.description if skill.description else "No skill description."
                        skill_line = f"- Skill: {skill.name if skill.name else 'Unnamed skill'} ({skill_desc})"
                        if skill.examples:
                            skill_line += f" (e.g., \"{skill.examples[0]}\")"
                        skills_list.append(skill_line)
                    
                    if skills_list:
                        skills_info = "\n    Skills:\n      " + "\n      ".join(skills_list)
                agent_descriptions.append(f"Agent Name: {name}\n  Description: {description}{skills_info}")
        
        available_agents_str = "\n\n".join(agent_descriptions) if agent_descriptions else "No remote agents currently available. You may need to use 'list_remote_agents' if this seems incorrect."

        return f"""You are an expert AI task router and delegator.
Your primary function is to understand the user's current request and delegate it to the most appropriate specialized remote agent.

Available Remote Agents and their Capabilities:
-------------------------------------------------
{available_agents_str}
-------------------------------------------------

Your Task:
1. Analyze the user's **current and most recent request** carefully.
2. Review the capabilities (name, description, and skills) of the available remote agents listed above.
3. Determine which single agent is **best suited** to handle this specific request.
4. **Crucially, you MUST use the 'send_task' tool to delegate the user's original request to the chosen agent.**
   - You need to provide the exact 'agent_name' of the chosen agent to the 'send_task' tool.
   - You need to provide the original 'message' (the user's full request) to the 'send_task' tool.
5. If no agent is suitable for the request, you should inform the user clearly that you cannot find a suitable agent to handle their specific request and why. Do not attempt to answer it yourself.
6. If multiple agents seem potentially relevant, choose the one that is most specialized for the core task in the user's request.
7. Do not attempt to answer the user's request directly. Your sole responsibility is to route the request to the correct agent using the 'send_task' tool or state that no agent is suitable.
8. After successfully initiating the task with `send_task`, you can inform the user which agent is handling their request, for example: "I am routing your request to the [Chosen Agent Name]..."

Tool Reminder:
- `send_task(agent_name: str, message: str)`: Use this tool to send the user's request to the selected agent.

Example Thought Process:
User Request: "write an article about how good classical music is for cows producing milk"
1. Analyze: The user wants an article written (text generation).
2. Review Agents: Look for an agent skilled in text generation, writing articles, or creative content.
   - If 'Text Generator Agent' (Description: Creates articles and textual content) exists: This is the best match.
   - If 'Currency Agent' (Description: Handles currency conversions) exists: This is not a match.
3. Determine: 'Text Generator Agent' is the one.
4. Use Tool: Call `send_task(agent_name="Text Generator Agent", message="write an article about how good classical music is for cows producing milk")`.
5. Inform User: "I am routing your request to the Text Generator Agent."

Now, process the user's actual request based on these instructions.
"""

    def check_state(self, context: ReadonlyContext):
        state = context.state
        if (
            'session_id' in state
            and 'session_active' in state
            and state['session_active']
            and 'agent' in state
        ):
            return {'active_agent': f'{state["agent"]}'}
        return {'active_agent': 'None'}

    def before_model_callback(
        self, callback_context: CallbackContext, llm_request
    ):
        state = callback_context.state
        if 'session_active' not in state or not state['session_active']:
            if 'session_id' not in state:
                state['session_id'] = str(uuid.uuid4())
            state['session_active'] = True

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.remote_agent_connections:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            remote_agent_info.append(
                {'name': card.name, 'description': card.description}
            )
        return remote_agent_info

    async def send_task(
        self, agent_name: str, message: str, tool_context: ToolContext
    ):
        """Sends a task either streaming (if supported) or non-streaming.

        This will send a message to the remote agent named agent_name.

        Args:
          agent_name: The name of the agent to send the task to.
          message: The message to send to the agent for the task.
          tool_context: The tool context this method runs in.

        Yields:
          A dictionary of JSON data.
        """
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f'Agent {agent_name} not found')
        state = tool_context.state
        state['agent'] = agent_name
        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f'Client not available for {agent_name}')
        if 'task_id' in state:
            taskId = state['task_id']
        else:
            taskId = str(uuid.uuid4())
        sessionId = state['session_id']
        task: Task
        messageId = ''
        metadata = {}
        if 'input_message_metadata' in state:
            metadata.update(**state['input_message_metadata'])
            if 'message_id' in state['input_message_metadata']:
                messageId = state['input_message_metadata']['message_id']
        if not messageId:
            messageId = str(uuid.uuid4())
        metadata.update(conversation_id=sessionId, message_id=messageId)
        request: TaskSendParams = TaskSendParams(
            id=taskId,
            sessionId=sessionId,
            message=Message(
                role='user',
                parts=[TextPart(text=message)],
                metadata=metadata,
            ),
            acceptedOutputModes=['text', 'text/plain', 'image/png'],
            # pushNotification=None,
            metadata={'conversation_id': sessionId},
        )
        task = await client.send_task(request, self.task_callback)
        # Assume completion unless a state returns that isn't complete
        state['session_active'] = task.status.state not in [
            TaskState.COMPLETED,
            TaskState.CANCELED,
            TaskState.FAILED,
            TaskState.UNKNOWN,
        ]
        if task.status.state == TaskState.INPUT_REQUIRED:
            # Force user input back
            tool_context.actions.skip_summarization = True
            tool_context.actions.escalate = True
        elif task.status.state == TaskState.CANCELED:
            # Open question, should we return some info for cancellation instead
            raise ValueError(f'Agent {agent_name} task {task.id} is cancelled')
        elif task.status.state == TaskState.FAILED:
            # Raise error for failure
            raise ValueError(f'Agent {agent_name} task {task.id} failed')
        response = []
        if task.status.message:
            # Assume the information is in the task message.
            response.extend(
                convert_parts(task.status.message.parts, tool_context)
            )
        if task.artifacts:
            for artifact in task.artifacts:
                response.extend(convert_parts(artifact.parts, tool_context))
        return response

    async def process_prompt_and_delegate(self, user_prompt: str, session_id: str) -> str:
        """
        Processes the user's prompt, uses the HostAgent's LLM to decide which sub-agent to delegate to,
        and then uses the 'send_task' tool to delegate.
        Returns a summary of the action taken or result.
        """
        if not hasattr(self, '_adk_agent_instance') or not self._adk_agent_instance:
            # Store the created ADK agent instance on self for potential reuse
            self._adk_agent_instance = self.create_agent()

        current_state = {'session_id': session_id} 

        class MockReadonlyContext(ReadonlyContext):
            def __init__(self, state_dict):
                self._state = state_dict
            @property
            def state(self): return self._state
            @property
            def history(self): return [] 
            @property
            def last_tool_metadata(self): return None # Corrected typo

        instruction_context = MockReadonlyContext(current_state)
        system_instruction_str = self.root_instruction(instruction_context)

        conversation_history = [
            types.Content(parts=[types.Part(text=user_prompt)], role="user")
        ]
        
        try:
            genai_model = self._adk_agent_instance._model 
            
            # Correctly access the tools in the format GenAI SDK expects
            # Assuming ADK Agent prepares tools in its tool_config.function_declarations
            if not hasattr(self._adk_agent_instance, 'tool_config') or \
               not hasattr(self._adk_agent_instance.tool_config, 'function_declarations'):
                # Fallback or error if tools are not found as expected
                # This might indicate a different ADK structure.
                # For now, we'll try to proceed with an empty list if not found,
                # though this would mean the LLM can't call tools.
                # A better approach would be to ensure tool_config is correctly populated by ADK.
                # print("Warning: Tool configuration not found in expected structure.", file=sys.stderr)
                genai_tools_list = []
            else:
                genai_tools_list = self._adk_agent_instance.tool_config.function_declarations

            response = await genai_model.generate_content_async(
                contents=conversation_history,
                generation_config=self._adk_agent_instance._generation_config,
                safety_settings=self._adk_agent_instance._safety_settings,
                tools=genai_tools_list, # Pass the list of GenAI Tool objects
                system_instruction=types.Content(parts=[types.Part(text=system_instruction_str)], role="system")
            )

            llm_response_content = ""
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        llm_response_content += part.text
                    elif part.function_call:
                        function_name = part.function_call.name
                        args = dict(part.function_call.args)
                        
                        if function_name == "send_task":
                            class MockToolContext(ToolContext):
                                def __init__(self, state_dict, adk_agent):
                                    self._state = state_dict
                                    self._adk_agent = adk_agent # The ADK Agent instance
                                    self.actions = self._MockToolContextActions()

                                @property
                                def state(self): return self._state
                                @property
                                def history(self): return []
                                def save_artifact(self, name: str, artifact_content: any): pass
                                def get_config(self) -> dict[str, any]: return {}
                                # Required by ADK ToolContext interface
                                @property
                                def agent(self): return self._adk_agent 

                                class _MockToolContextActions:
                                    def __init__(self):
                                        self.skip_summarization = False
                                        self.escalate = False
                                        self.respond_to_user = False
                                        self.end_session = False
                                        self.auth_error = False

                            tool_ctx = MockToolContext(current_state, self._adk_agent_instance)
                            
                            if 'message' not in args:
                                args['message'] = user_prompt 
                            
                            tool_result = await self.send_task(
                                agent_name=args['agent_name'], 
                                message=args['message'], 
                                tool_context=tool_ctx
                            )
                            if isinstance(tool_result, list):
                                result_str = " ".join(str(r) for r in tool_result if r)
                            else:
                                result_str = str(tool_result)

                            confirmation_message = llm_response_content if llm_response_content else f"Task sent to {args['agent_name']}."
                            return f"{confirmation_message}\nSub-agent response/status: {result_str if result_str else 'No immediate data.'}"
                        else:
                            return f"Host LLM tried to call unhandled tool: {function_name}"
            
            if llm_response_content:
                return f"HostAgent LLM response: {llm_response_content}"
            else:
                # Check if there was a blocked prompt or other issue
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    return f"HostAgent LLM request was blocked: {response.prompt_feedback.block_reason_message}"
                return "HostAgent LLM did not provide a response or a recognized tool call."

        except Exception as e:
            # Ensure traceback is imported if used here. It was added in the previous apply.
            return f"Error in HostAgent processing: {str(e)}\n{traceback.format_exc()}"


def convert_parts(parts: list[Part], tool_context: ToolContext):
    rval = []
    for p in parts:
        rval.append(convert_part(p, tool_context))
    return rval


def convert_part(part: Part, tool_context: ToolContext):
    if part.type == 'text':
        return part.text
    if part.type == 'data':
        return part.data
    if part.type == 'file':
        # Repackage A2A FilePart to google.genai Blob
        # Currently not considering plain text as files
        file_id = part.file.name
        file_bytes = base64.b64decode(part.file.bytes)
        file_part = types.Part(
            inline_data=types.Blob(
                mime_type=part.file.mimeType, data=file_bytes
            )
        )
        tool_context.save_artifact(file_id, file_part)
        tool_context.actions.skip_summarization = True
        tool_context.actions.escalate = True
        return DataPart(data={'artifact-file-id': file_id})
    return f'Unknown type: {part.type}'
