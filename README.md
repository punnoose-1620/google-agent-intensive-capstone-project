# Educational Tutor Agent - Project Structure

> **Note:** This project is a capstone project for the **AI Agents Intensive Course with Google** offered on Kaggle.

This document explains the file structure of the Educational Tutor Agent project.

## Project Overview

The Educational Tutor Agent is a multi-agent system designed to provide personalized educational tutoring. The project is organized into a modular structure that separates core framework components, specialized agents, tools, and demonstration notebooks.

## Directory Structure

```
educational-tutor-agent/
├── notebooks/              # Jupyter notebooks for demonstrations and setup
├── src/                    # Core source code modules
│   ├── agents/            # Specialized agent implementations
│   └── tools/             # Utility tools for agents
├── tests/                 # Unit and integration tests
├── data/                  # Persistent data storage
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Detailed File Structure

### `/notebooks/` - Demonstration Notebooks

Contains Jupyter notebooks for setup, demonstrations, and end-to-end workflows:

- **`00_setup.ipynb`** - Initial setup and environment configuration
  - Installs project dependencies from `requirements.txt`
  - Displays current working directory and project root
  - Visualizes project structure in tree format
  - Loads environment variables from `.env` file with proper path detection
  - Validates `GEMINI_API_KEY` is configured
- **`01_agents_demo.ipynb`** - Demonstration of individual agent capabilities
- **`02_end_to_end_demo.ipynb`** - Complete end-to-end workflow demonstration

### `/src/` - Core Source Code

The main source code directory containing the framework and implementations:

#### Core Framework Files

- **`agent_framework.py`** - Base agent framework including the `Agent` base class and `Coordinator` for agent orchestration
- **`config.py`** - Configuration management for API keys, settings, and environment variables
- **`memory.py`** - Memory management system for session storage and long-term persistence
- **`observability.py`** - Logging, metrics, and tracing utilities for monitoring agent interactions

#### `/src/agents/` - Specialized Agents

Contains implementations of specialized agents:

- **`search_agent.py`** - Agent responsible for web search and resource discovery
- **`tutor_agent.py`** - Core tutoring agent that provides explanations and educational content
- **`quiz_agent.py`** - Agent for generating quizzes and grading student answers
- **`evaluator_agent.py`** - Agent that evaluates explanations and quizzes using LLM-as-judge methodology

#### `/src/tools/` - Utility Tools

Tools that agents can use to interact with external resources:

- **`web_search.py`** - Web search functionality for finding educational resources
- **`pdf_fetcher.py`** - PDF fetching and parsing utilities
- **`code_executor.py`** - Safe code execution environment for running code examples

### `/tests/` - Test Suite

Unit and integration tests:

- **`test_agents.py`** - Tests for agent functionality and coordination
- **`test_memory.py`** - Tests for memory storage and retrieval

### `/data/` - Data Storage

Persistent data directory:

- **`memory_store.json`** - JSON-based storage for long-term memory (user progress, preferences, etc.)

### Root Files

- **`requirements.txt`** - Python package dependencies
- **`README.md`** - Project documentation (this file)
- **`LICENSE`** - License file
- **`.gitignore`** - Git ignore patterns

## Module Organization

The project follows a modular architecture:

1. **Framework Layer** (`agent_framework.py`) - Provides the base infrastructure for agent communication
2. **Agent Layer** (`agents/`) - Specialized agents that handle specific educational tasks
3. **Tool Layer** (`tools/`) - Reusable utilities that agents can leverage
4. **Infrastructure Layer** (`memory.py`, `observability.py`, `config.py`) - Supporting systems for persistence, monitoring, and configuration

## Usage Flow

1. **Setup** - Run `00_setup.ipynb` to:
   - Install all required dependencies
   - Verify project structure and working directory
   - Load and validate environment variables (`.env` file)
   - Ensure `GEMINI_API_KEY` is properly configured
2. **Development** - Use `01_agents_demo.ipynb` to test individual components
3. **Demonstration** - Run `02_end_to_end_demo.ipynb` for a complete workflow example

## Setup Details

The `00_setup.ipynb` notebook performs the following operations:

1. **Dependency Installation**: Installs all packages listed in `requirements.txt`
2. **Directory Detection**: Automatically detects the current working directory and project root (handles both running from project root and `notebooks/` subdirectory)
3. **Project Structure Visualization**: Displays the project directory tree structure up to 3 levels deep
4. **Environment Configuration**: 
   - Locates the `.env` file in the project root
   - Loads environment variables using `python-dotenv`
   - Handles different file encodings (UTF-8, UTF-16) for cross-platform compatibility
   - Validates that `GEMINI_API_KEY` is present and configured

## Dependencies

Install dependencies using:
```bash
pip install -r requirements.txt
```

Or run the setup notebook which will install them automatically:
```bash
jupyter notebook notebooks/00_setup.ipynb
```

## Environment Configuration

Ensure you have a `.env` file in the project root with your `GEMINI_API_KEY` configured:

```env
GEMINI_API_KEY=your_api_key_here
```

**Note**: The setup notebook automatically detects the project root and loads the `.env` file. If you encounter encoding issues (particularly on Windows), the notebook handles UTF-16 encoding automatically. For best compatibility, save your `.env` file as UTF-8 encoding.

