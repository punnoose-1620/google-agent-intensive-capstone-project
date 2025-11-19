"""
Prompt Templates - Reusable prompt templates for agents.

This module contains prompt templates for various agent actions.
Each template function returns a formatted string ready to send to the LLM.
"""

from typing import Optional


def TUTOR_EXPLAIN_PROMPT(topic: str, level: str = "intermediate", additional_context: Optional[str] = None) -> str:
    """
    Generate a prompt for explaining a topic at a specific level.
    
    Best Practices:
    - Use temperature=0.0 for deterministic, reproducible explanations
    - Use max_tokens=1024 for comprehensive explanations
    - Structure the prompt with clear role, instruction, and content sections
    - Request JSON output for easy parsing
    
    Args:
        topic: Topic to explain
        level: Difficulty level (beginner/intermediate/advanced)
        additional_context: Optional additional context or requirements
        
    Returns:
        Formatted prompt string
        
    Example:
        >>> prompt = TUTOR_EXPLAIN_PROMPT("neural networks", "beginner")
        >>> # Use with: model.generate_content(prompt, temperature=0.0, max_output_tokens=1024)
    """
    context_section = f"\nAdditional context: {additional_context}" if additional_context else ""
    
    prompt = f"""You are an expert university tutor. Your role is to provide clear, structured explanations.

Role: tutor
Instruction: Explain the topic "{topic}" at {level} level. Provide a structured explanation with summary, step-by-step breakdown, examples, and potential difficulties.
{context_section}

Return your response as valid JSON only with the following structure:
{{
    "summary": "A 2-3 sentence overview of the topic",
    "step_by_step": ["Step 1 description", "Step 2 description", ...],
    "examples": [
        {{
            "title": "Example 1 title",
            "description": "Example description",
            "solution": "Solution or explanation"
        }}
    ],
    "key_equations": ["Equation 1", "Equation 2", ...],
    "difficulties": ["Common difficulty 1", "Common difficulty 2", ...],
    "further_reading": [
        {{
            "title": "Resource title",
            "url": "Resource URL or description"
        }}
    ],
    "confidence": "low|medium|high"
}}

Do not include any markdown formatting, code blocks, or extra commentary outside the JSON.
Return only valid JSON.

Recommended settings:
- temperature: 0.0 (for deterministic, consistent explanations)
- max_tokens: 1024 (for comprehensive explanations)"""
    
    return prompt


def QUIZ_GENERATE_PROMPT(
    topic: str, 
    difficulty: str = "intermediate", 
    n_questions: int = 5,
    question_type: str = "multiple_choice"
) -> str:
    """
    Generate a prompt for creating quiz questions on a topic.
    
    Best Practices:
    - Use temperature=0.3-0.5 for creative but consistent question generation
    - Use max_tokens=1024 for multiple questions with explanations
    - Request structured JSON output with questions, options, answers, and explanations
    - Include difficulty-appropriate question complexity
    
    Args:
        topic: Topic for quiz questions
        difficulty: Difficulty level (beginner/intermediate/advanced)
        n_questions: Number of questions to generate (default: 5)
        question_type: Type of questions (multiple_choice/short_answer/both)
        
    Returns:
        Formatted prompt string
        
    Example:
        >>> prompt = QUIZ_GENERATE_PROMPT("linear regression", "intermediate", n_questions=5)
        >>> # Use with: model.generate_content(prompt, temperature=0.4, max_output_tokens=1024)
    """
    question_type_desc = {
        "multiple_choice": "multiple-choice questions with 4 options each",
        "short_answer": "short-answer questions",
        "both": "a mix of multiple-choice and short-answer questions"
    }.get(question_type, "multiple-choice questions")
    
    prompt = f"""You are an expert educator creating quiz questions.

Role: quiz_generator
Instruction: Generate {n_questions} {question_type_desc} on the topic "{topic}" at {difficulty} difficulty level.

Requirements:
- Questions should test understanding of key concepts
- Difficulty should be appropriate for {difficulty} level
- Include clear, unambiguous questions
- Provide correct answers with explanations
- For multiple-choice: include 4 options with one clearly correct answer

Return your response as valid JSON only with the following structure:
{{
    "questions": [
        {{
            "id": "q1",
            "type": "multiple_choice" or "short_answer",
            "question": "Question text",
            "options": ["Option A", "Option B", "Option C", "Option D"],  // Only for multiple_choice
            "correct_answer": "Correct answer or option index",
            "answer_index": 0,  // 0-based index for multiple_choice (0=A, 1=B, etc.)
            "explanation": "Explanation of why this is correct",
            "points": 1
        }}
    ],
    "total_points": {n_questions},
    "estimated_time_minutes": 10
}}

Do not include any markdown formatting, code blocks, or extra commentary outside the JSON.
Return only valid JSON.

Recommended settings:
- temperature: 0.4 (balances creativity with consistency)
- max_tokens: 1024 (for {n_questions} questions with explanations)"""
    
    return prompt


def EVALUATOR_PROMPT(source_text: str, candidate: str, evaluation_criteria: Optional[str] = None) -> str:
    """
    Generate a prompt for evaluating candidate text against source text.
    
    Best Practices:
    - Use temperature=0.0 for consistent, reproducible evaluations
    - Use max_tokens=512 for detailed evaluation scores
    - Request structured scoring with specific criteria
    - Ask for evidence-based evaluation with citations
    
    This is used for LLM-as-judge evaluation of explanations, summaries, etc.
    
    Args:
        source_text: Original source text or ground truth
        candidate: Candidate text to evaluate (explanation, summary, etc.)
        evaluation_criteria: Optional specific criteria to evaluate (accuracy, clarity, etc.)
        
    Returns:
        Formatted prompt string
        
    Example:
        >>> prompt = EVALUATOR_PROMPT(source_text, candidate_summary)
        >>> # Use with: model.generate_content(prompt, temperature=0.0, max_output_tokens=512)
    """
    criteria_section = ""
    if evaluation_criteria:
        criteria_section = f"\nFocus on these criteria: {evaluation_criteria}"
    else:
        criteria_section = "\nEvaluate on: accuracy, clarity, completeness, and usefulness"
    
    prompt = f"""You are an expert evaluator assessing the quality of educational content.

Role: evaluator
Instruction: Evaluate the candidate text against the source text. Assess accuracy, clarity, completeness, usefulness, and identify any hallucinations or incorrect information.{criteria_section}

Source Text:
{source_text}

Candidate Text to Evaluate:
{candidate}

Return your response as valid JSON only with the following structure:
{{
    "accuracy_score": 1-5,  // 1=very inaccurate, 5=completely accurate
    "clarity_score": 1-5,   // 1=very unclear, 5=very clear
    "completeness_score": 1-5,  // 1=very incomplete, 5=completely covers topic
    "usefulness_score": 1-5,   // 1=not useful, 5=very useful
    "overall_score": 1-5,  // Overall quality score
    "hallucinated_claims": [
        {{
            "claim": "Specific claim from candidate text",
            "issue": "Why this is problematic (not in source, contradicts source, etc.)",
            "severity": "low|medium|high"
        }}
    ],
    "missing_key_points": ["Key point 1 from source not in candidate", ...],
    "strengths": ["Strength 1", "Strength 2", ...],
    "weaknesses": ["Weakness 1", "Weakness 2", ...],
    "recommendations": ["Recommendation 1", "Recommendation 2", ...]
}}

For each hallucinated claim, quote the exact problematic sentence from the candidate text.
If a claim cannot be verified in the source text, mark it as a hallucination.

Do not include any markdown formatting, code blocks, or extra commentary outside the JSON.
Return only valid JSON.

Recommended settings:
- temperature: 0.0 (for deterministic, consistent evaluations)
- max_tokens: 512 (for detailed evaluation with evidence)"""
    
    return prompt


def TUTOR_SUMMARIZE_NOTES_PROMPT(notes: str, focus_areas: Optional[list] = None) -> str:
    """
    Generate a prompt for summarizing learning notes.
    
    Best Practices:
    - Use temperature=0.0 for consistent summarization
    - Use max_tokens=512 for concise summaries
    - Request structured output with key concepts and action items
    
    Args:
        notes: Notes to summarize (string or will be converted from list)
        focus_areas: Optional list of specific areas to focus on
        
    Returns:
        Formatted prompt string
    """
    if isinstance(notes, list):
        notes = "\n".join([str(note) for note in notes])
    
    focus_text = f"\nFocus on these areas: {', '.join(focus_areas)}" if focus_areas else ""
    
    prompt = f"""You are an expert tutor helping to summarize learning notes.

Role: tutor
Instruction: Summarize the following learning notes into key concepts, important points, and action items.{focus_text}

Notes:
{notes}

Return your response as valid JSON only:
{{
    "summary": "Brief overall summary",
    "key_concepts": ["Concept 1", "Concept 2", ...],
    "important_points": ["Point 1", "Point 2", ...],
    "action_items": ["Action 1", "Action 2", ...],
    "gaps": ["Knowledge gap 1", "Knowledge gap 2", ...]
}}

Return only valid JSON.

Recommended settings:
- temperature: 0.0 (for consistent summarization)
- max_tokens: 512 (for concise summaries)"""
    
    return prompt


def TUTOR_CREATE_EXAMPLE_PROMPT(
    topic: str, 
    difficulty: str = "intermediate", 
    example_type: str = "problem"
) -> str:
    """
    Generate a prompt for creating educational examples.
    
    Best Practices:
    - Use temperature=0.3-0.5 for creative but educational examples
    - Use max_tokens=512 for detailed examples with solutions
    - Request structured output with problem, solution, and explanation
    
    Args:
        topic: Topic for the example
        difficulty: Difficulty level (beginner/intermediate/advanced)
        example_type: Type of example (problem/demonstration/case_study)
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert tutor creating educational examples.

Role: tutor
Instruction: Create a {example_type} example for the topic "{topic}" at {difficulty} difficulty level.

Requirements:
- Example should clearly demonstrate key concepts
- Difficulty should match {difficulty} level
- Include step-by-step solution or explanation
- Make it educational and illustrative

Return your response as valid JSON only:
{{
    "title": "Example title",
    "problem": "Problem statement or question",
    "solution": "Step-by-step solution",
    "explanation": "Why this solution works",
    "key_concepts": ["Concept 1", "Concept 2", ...],
    "variations": ["Variation 1", "Variation 2", ...]
}}

Return only valid JSON.

Recommended settings:
- temperature: 0.3 (for creative but educational examples)
- max_tokens: 512 (for detailed examples)"""
    
    return prompt


def TUTOR_ADAPT_TO_USER_PROMPT(
    content: str,
    user_level: str = "intermediate",
    user_preferences: Optional[dict] = None,
    user_history: Optional[list] = None
) -> str:
    """
    Generate a prompt for adapting content to a user's level and preferences.
    
    Best Practices:
    - Use temperature=0.0 for consistent adaptation
    - Use max_tokens=1024 for comprehensive adaptations
    - Consider user's learning style and history
    
    Args:
        content: Content to adapt
        user_level: User's current level (beginner/intermediate/advanced)
        user_preferences: Optional user preferences (learning_style, etc.)
        user_history: Optional list of previous topics or difficulties
        
    Returns:
        Formatted prompt string
    """
    preferences_text = ""
    if user_preferences:
        preferences_text = f"\nUser preferences: {', '.join([f'{k}: {v}' for k, v in user_preferences.items()])}"
    
    history_text = ""
    if user_history:
        history_text = f"\nUser learning history: {', '.join(user_history)}"
    
    prompt = f"""You are an expert tutor adapting content to a specific learner.

Role: tutor
Instruction: Adapt the following content to match the user's level ({user_level}) and preferences.

Content to adapt:
{content}
{preferences_text}
{history_text}

Return your response as valid JSON only:
{{
    "adapted_content": "Content adapted to user's level and preferences",
    "simplifications": ["Simplification 1", "Simplification 2", ...],
    "enhancements": ["Enhancement 1", "Enhancement 2", ...],
    "personalized_examples": ["Example 1", "Example 2", ...],
    "recommended_next_steps": ["Step 1", "Step 2", ...]
}}

Return only valid JSON.

Recommended settings:
- temperature: 0.0 (for consistent adaptation)
- max_tokens: 1024 (for comprehensive adaptations)"""
    
    return prompt

