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

## Notebook Results and Demonstrations

### `01_agents_demo.ipynb` - Individual Agent Capabilities

This notebook successfully demonstrates the multi-agent system with the following results:

#### ✅ Successful Demonstrations

1. **TutorAgent - Explanation Generation**
   - Successfully generated structured explanation for "bias-variance tradeoff" at intermediate level
   - Produced comprehensive content including:
     - Summary: Clear 2-3 sentence overview
     - Step-by-step breakdown: 5 detailed steps explaining the concept
     - Examples: 2 practical examples (Linear Regression vs. Polynomial, Decision Tree Depth)
     - Key equations: "Total Error = Bias² + Variance + Irreducible Error"
     - Difficulties: 4 common learning challenges identified
     - Further reading: 2 educational resources with URLs
     - Confidence level: "high"

2. **QuizAgent - Quiz Generation**
   - Successfully generated 5 quiz questions on the topic
   - Each question includes:
     - Multiple choice options (A, B, C, D)
     - Correct answer identification
     - Detailed explanations

3. **EvaluatorAgent - Quality Assessment**
   - Successfully evaluated explanation quality using LLM-as-judge methodology
   - Scores provided:
     - Accuracy: 4/5
     - Clarity: 5/5
     - Completeness: 4/5
     - Usefulness: 5/5
     - Overall: 4.5/5
   - Identified strengths and weaknesses
   - No hallucinations detected

4. **Memory Storage**
   - Successfully saved all results (explanation, quiz, evaluation) to persistent memory
   - Tag-based search functionality working correctly
   - Session continuity demonstrated

#### Key Features Demonstrated
- ✅ Multi-agent coordination via Coordinator
- ✅ Structured message passing between agents
- ✅ LLM-powered content generation (Gemini API)
- ✅ Automatic quality assessment
- ✅ Persistent memory storage with tagging

### `02_end_to_end_demo.ipynb` - Complete Workflow

This notebook demonstrates a full end-to-end workflow on "Bayes theorem" at intermediate level.

#### ⚠️ Known Issue: JSON Parsing Error

**Current Status**: The notebook encounters a JSON parsing error when processing TutorAgent responses.

**Error Details**:
```
Failed to parse JSON from response: Expecting ',' delimiter: line 16 column 10 (char 2180)
Response: {
    "summary": "Bayes' Theorem is a fundamental concept in probability theory that describes how to update the probability of a hypothesis based on new evidence. It allows us to revise our beliefs i
```

**Root Cause**:
- The Gemini API response is being truncated mid-word ("beliefs i" instead of complete text)
- The response exceeds the `max_tokens` limit (currently set to 8192)
- The JSON extraction logic attempts to parse incomplete JSON, causing a parsing error
- Manual brace counting in `_extract_json_from_response()` may not correctly handle all edge cases

**Error Location**:
- File: `src/agents/tutor_agent.py`
- Method: `_extract_json_from_response()` (lines 191-284)
- Issue: Response truncation detection and JSON extraction from incomplete responses

**Attempted Solutions**:
1. ✅ Increased `max_tokens` from 2048 to 8192 - Issue persists
2. ✅ Added truncation detection via `finish_reason == 'MAX_TOKENS'` - Not always triggered
3. ⚠️ Manual brace counting for JSON extraction - May have edge cases

**Recommended Fix**:
- Replace manual brace counting with Python's built-in `json.JSONDecoder.raw_decode()` method
- This automatically handles nested objects, arrays, escaped characters, and all JSON edge cases
- Better detection of truncation by checking if parsing error occurs near the end of response
- More robust error messages with position information

**Workaround**:
- Use simpler topics that generate shorter responses
- Manually increase `max_tokens` further (though 8192 should be sufficient)
- Simplify the prompt to request shorter explanations

#### Expected Workflow (When Fixed)

The complete workflow should demonstrate:

1. **Step 1: Explanation Generation** - TutorAgent explains "Bayes theorem"
2. **Step 2: Quiz Generation** - QuizAgent creates assessment questions
3. **Step 3: Quality Evaluation** - EvaluatorAgent evaluates explanation quality
4. **Step 4: Memory Storage** - All results saved for future reference
5. **Step 5: Quiz Grading** - Demonstrate answer grading and adaptive learning
6. **Step 6: Observability** - View metrics and logs from the session

## Known Issues and Troubleshooting

### JSON Parsing Error with Truncated Responses

**Issue**: TutorAgent responses are sometimes truncated, causing JSON parsing failures.

**Symptoms**:
- Error: `Failed to parse JSON from response: Expecting ',' delimiter`
- Response text cuts off mid-word
- JSON appears incomplete (unclosed braces)

**Current Status**: Under investigation. The issue occurs when:
- Response length exceeds token limits
- `finish_reason` check doesn't catch truncation
- Manual brace counting in JSON extraction has edge cases

**Temporary Workarounds**:
1. Use simpler topics that generate shorter responses
2. Increase `max_tokens` parameter (currently 8192)
3. Simplify prompts to request more concise outputs

**Planned Fix**: Replace manual JSON extraction with `json.JSONDecoder.raw_decode()` for more robust parsing.

### Environment Variable Encoding

**Issue**: `.env` file encoding issues on Windows.

**Solution**: The setup notebook automatically handles UTF-16 encoding. For best results, save `.env` as UTF-8.

## Contributing

When working on this project:

1. Run `00_setup.ipynb` first to ensure environment is configured
2. Test individual components with `01_agents_demo.ipynb`
3. Run full workflow with `02_end_to_end_demo.ipynb` (note: currently has JSON parsing issue)
4. Check logs in `data/logs/` for debugging information
5. Review observability metrics for agent interactions

