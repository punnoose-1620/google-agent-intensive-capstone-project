# Title
Educational Tutor Agent — Multi-Agent AI Tutor with Quiz & Evaluation

# Subtitle
Personalized, multi-agent tutoring system powered by Gemini, offering explanations, adaptive quizzes, and automated evaluation with persistent memory and observability.

# Track
Agents for Good (Education)

# Project Description (<1500 words)

## Problem (Why this matters)
Students often struggle to learn complex concepts because explanations are generic, practice is non-adaptive, and progress isn’t tracked. Teachers can’t always provide instant, personalized feedback. An AI tutoring assistant that explains topics, generates targeted quizzes, and evaluates learning quality can fill that gap at scale.

## Solution (What I built)
I built a multi-agent Educational Tutor Agent system with the following components:

- **Coordinator** — central orchestrator handling message flow, timeouts, and observability hooks.

- **Tutor Agent** — generates structured, leveled explanations (summary, step-by-step breakdown, examples, key equations, difficulties) using Gemini API with retry logic.

- **Quiz Agent** — creates multiple-choice quizzes and grades student responses, appending incorrect answers to memory for adaptive learning.

- **Evaluator Agent** — uses LLM-as-judge methodology with Gemini (temperature=0) to score explanations and quizzes on accuracy, clarity, usefulness, and hallucination detection.

- **Memory** — JSON/TinyDB backed long-term memory to store user profiles, weak points, past sessions, and tag-based search functionality.

- **Observability** — logs, traces and simple metrics (latency, errors, event counts) for debugging and quality assurance.

The system includes custom tools (web search placeholder, PDF fetcher, safe code executor) that agents can use. Gemini powers the Tutor and Evaluator agents for high-quality natural language generation. The system demonstrates sequential agent workflows, long-term memory, and observability.

## Architecture (How it works)

User → Coordinator → TutorAgent → QuizAgent → EvaluatorAgent → Memory.

The Coordinator orchestrates message flow between agents. Each agent handles specific actions:
- **TutorAgent**: Processes "explain" actions to generate structured educational content
- **QuizAgent**: Handles "generate_quiz" and "grade_answer" actions
- **EvaluatorAgent**: Performs "evaluate_summary" and "evaluate_quiz" actions using LLM-as-judge

**Key design choices:**

- Modular agents for separation of concerns and easier testing.

- Deterministic prompts (temperature=0) for evaluation tasks to reduce variability.

- Retry logic with exponential backoff for transient API errors.

- Atomic memory writes and tag-based indexing for safe persistence.

- Robust JSON extraction from LLM responses using json.JSONDecoder.raw_decode().

(Architecture diagram included in attachments: coverImage1.png)

**Implementation highlights**

- ´src/agent_framework.py´ (Agent base, Coordinator)

- ´src/agents/´ (tutor_agent.py, quiz_agent.py, evaluator_agent.py, prompts.py)

- ´src/tools/´ (web_search.py, pdf_fetcher.py, code_executor.py)

- ´src/memory.py´ (TinyDB/JSON memory store)

- ´src/observability.py´ (loguru logs + metrics counters)

**Notebooks**: 00_setup.ipynb, 01_agents_demo.ipynb, 02_end_to_end_demo.ipynb (demo flows)

# Results & Demonstrations

In **01_agents_demo.ipynb**:

- TutorAgent produced structured explanations (summary + 5 step breakdown + 2 examples) for “bias-variance tradeoff” with high clarity.

- QuizAgent generated 5 multiple-choice questions with correct answers and explanations.

- EvaluatorAgent scored the generated explanation: Accuracy 4/5, Clarity 5/5, Usefulness 5/5.

In **02_end_to_end_demo.ipynb**:

- Full pipeline demonstrated for "Bayes' theorem" (explain → quiz → evaluate → memory persist). JSON parsing improvements have been implemented to handle long responses (see Known Issues for details).

# Evaluation & Observability

- Automatic scoring by EvaluatorAgent (accuracy/clarity/usefulness) using Gemini (temperature=0).

- Observability includes request-level traces, logs saved as JSONL format, and counters for agent events (message_sent, message_handled, errors). Logs are written to data/logs/{date}.log and sample traces are shown in the demo notebook.

# How to run (quickstart)

1. Clone repo: git clone https://github.com/<you>/educational-tutor-agent

2. Create .env (do NOT commit): GEMINI_API_KEY=your_key

3. Install: pip install -r requirements.txt

4. Run setup notebook: open notebooks/00_setup.ipynb and execute.

5. Try individual agents: notebooks/01_agents_demo.ipynb

6. Run full pipeline: notebooks/02_end_to_end_demo.ipynb

## What I learned & future work

- Robust JSON extraction from LLMs is essential — implemented json.JSONDecoder.raw_decode() to handle nested structures and detect truncation.

- Retry logic with exponential backoff is crucial for handling transient API errors (500, 503, 429).

- Improve robustness with multi-judge averaging and hybrid rule-based checks to reduce hallucination risk.

- Future: integrate SearchAgent for finding educational resources, add deployed demo (Agent Engine / Cloud Run) for +5 bonus, and create a short video walkthrough for +10.

## Known Issues

JSON parsing error when Tutor responses are truncated mid-token (occurs on very long prompts). This has been addressed by:
- Implementing json.JSONDecoder.raw_decode() for robust JSON extraction
- Adding explicit truncation detection via finish_reason == 'MAX_TOKENS'
- Retry logic with increased max_tokens when truncation is detected
- Workarounds: simplify prompts or increase max_tokens if issues persist

# Conclusion
This capstone demonstrates a complete, modular multi-agent tutoring system covering core course concepts: multi-agent orchestration, tools/MCP, sessions & memory, observability, and LLM-based evaluation. All code and notebooks are public in the linked GitHub repo and are ready for reviewers to run.