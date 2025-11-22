"""
Agents package - Specialized agent implementations.

This package contains implementations of specialized agents for the
Educational Tutor Agent system.
"""

from .tutor_agent import TutorAgent
from .evaluator_agent import EvaluatorAgent
from .quiz_agent import QuizAgent
from .prompts import (
    TUTOR_EXPLAIN_PROMPT,
    QUIZ_GENERATE_PROMPT,
    EVALUATOR_PROMPT,
    TUTOR_SUMMARIZE_NOTES_PROMPT,
    TUTOR_CREATE_EXAMPLE_PROMPT,
    TUTOR_ADAPT_TO_USER_PROMPT,
)

__all__ = [
    'TutorAgent',
    'EvaluatorAgent',
    'QuizAgent',
    'TUTOR_EXPLAIN_PROMPT',
    'QUIZ_GENERATE_PROMPT',
    'EVALUATOR_PROMPT',
    'TUTOR_SUMMARIZE_NOTES_PROMPT',
    'TUTOR_CREATE_EXAMPLE_PROMPT',
    'TUTOR_ADAPT_TO_USER_PROMPT',
]

