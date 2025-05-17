# AI Heroes 2025 - A2A Protocol Demo

This project demonstrates the implementation of Google's Agent-to-Agent (A2A) Protocol across different AI agent frameworks. It showcases how various types of agents can communicate and interoperate seamlessly using the A2A protocol.

## Project Components

### CrewAI Agent Server

Located in `app/agents/crewai`, this component implements an A2A server using the CrewAI framework. The agent demonstrates how CrewAI's capabilities can be exposed through the A2A protocol, allowing other agents to interact with its specialized functionalities.

### LangGraph Agent Server

Found in `app/agents/langgraph`, this implementation shows how LangGraph-based agents can serve as A2A protocol servers. It demonstrates the integration of LangGraph's powerful flow-based agent architecture with the A2A protocol's communication standards.

### LangChain Hybrid Agent

The `app/agents/langchain` component showcases a dual-role implementation:

- Acts as an A2A server, exposing LangChain agent capabilities
- Functions as an A2A client, able to consume services from other A2A-compatible agents

### CLI Client

A command-line interface that acts as an A2A client, allowing users to:

- Input prompts through terminal interaction
- Connect to and communicate with any of the available A2A server agents
- Receive and display responses from the agents

## Architecture Overview

The project demonstrates a complete A2A ecosystem where:

- Multiple agent frameworks (CrewAI, LangGraph, LangChain) expose their capabilities as A2A servers
- The LangChain implementation shows bi-directional A2A capabilities
- The CLI provides a user-friendly interface to interact with the entire system

This architecture showcases the power of the A2A protocol in enabling seamless communication between different AI agent frameworks, regardless of their underlying implementation.

## Getting Started

1. Each agent component has its own `pyproject.toml` for dependency management
2. Follow the individual README files in each component's directory for specific setup instructions
3. Use the CLI client to interact with the agents and observe the A2A protocol in action
