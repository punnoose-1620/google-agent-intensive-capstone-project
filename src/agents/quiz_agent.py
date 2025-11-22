"""
Quiz Agent - Generate quizzes and grade student answers.

This agent generates quiz questions and grades student responses,
tracking incorrect answers in memory for adaptive learning.
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
from difflib import SequenceMatcher

# Import base Agent class
from ..agent_framework import Agent

# Import prompt template and memory store
from .prompts import QUIZ_GENERATE_PROMPT
from ..memory import MemoryStore

# Try to import Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


class QuizAgent(Agent):
    """
    Quiz agent that generates questions and grades student answers.
    
    Uses Google Gemini API to generate quiz questions and grade responses.
    Tracks incorrect answers in memory for adaptive learning.
    """
    
    def __init__(
        self, 
        name: str = "quiz_agent", 
        api_key: Optional[str] = None, 
        model_name: str = "gemini-pro",
        memory_store: Optional[MemoryStore] = None
    ):
        """
        Initialize the quiz agent.
        
        Args:
            name: Agent name identifier
            api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
            model_name: Gemini model to use (default: "gemini-pro")
            memory_store: Optional MemoryStore instance for saving wrong answers
        """
        super().__init__(name=name)
        
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package is required. "
                "Install with: pip install google-generativeai"
            )
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in environment or pass as api_key parameter."
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        # Initialize memory store
        self.memory_store = memory_store or MemoryStore()
        
        # Store generated quizzes (in-memory cache)
        # Format: {quiz_id: {topic, questions: [...], ...}}
        self.quizzes: Dict[str, Dict[str, Any]] = {}
    
    def _call_gemini(
        self, 
        prompt: str, 
        max_tokens: int = 1024, 
        temperature: float = 0.4
    ) -> str:
        """
        Call Gemini API with the given prompt.
        
        Args:
            prompt: Prompt text to send to Gemini
            max_tokens: Maximum tokens in response (default: 1024)
            temperature: Sampling temperature (0.4 for quiz generation)
            
        Returns:
            Response text from Gemini
            
        Raises:
            Exception: If API call fails
        """
        try:
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Gemini API call failed: {str(e)}")
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response, handling markdown code blocks.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If JSON cannot be extracted or parsed
        """
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object in the text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from response: {e}\nResponse: {response_text[:200]}")
    
    def generate_quiz(self, topic: str, n_questions: int = 5, difficulty: str = "intermediate") -> Dict[str, Any]:
        """
        Generate a quiz on the given topic.
        
        Args:
            topic: Topic for the quiz
            n_questions: Number of questions to generate (default: 5)
            difficulty: Difficulty level (beginner/intermediate/advanced)
            
        Returns:
            Dictionary with quiz data including questions list
        """
        # Generate prompt
        prompt = QUIZ_GENERATE_PROMPT(topic, difficulty, n_questions, "multiple_choice")
        
        # Call Gemini API
        response_text = self._call_gemini(prompt, max_tokens=1024, temperature=0.4)
        quiz_data = self._extract_json_from_response(response_text)
        
        # Add topic and generate quiz ID
        import uuid
        quiz_id = f"quiz_{uuid.uuid4().hex[:8]}"
        quiz_data["quiz_id"] = quiz_id
        quiz_data["topic"] = topic
        quiz_data["difficulty"] = difficulty
        
        # Store quiz in memory
        self.quizzes[quiz_id] = quiz_data
        
        return quiz_data
    
    def grade_answer(
        self, 
        question_id: str, 
        student_answer: str,
        quiz_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Grade a student's answer to a question.
        
        Args:
            question_id: ID of the question being answered
            student_answer: Student's answer
            quiz_id: Optional quiz ID (if not provided, searches all quizzes)
            user_id: Optional user ID for saving incorrect answers to memory
            
        Returns:
            Dictionary with keys:
            - correct: bool - Whether answer is correct
            - score: float - Score (0.0 to 1.0)
            - explanation: str - Explanation of the answer
        """
        # Find the question
        question = None
        quiz_data = None
        
        if quiz_id and quiz_id in self.quizzes:
            quiz_data = self.quizzes[quiz_id]
            question = next((q for q in quiz_data.get("questions", []) if q.get("id") == question_id), None)
        else:
            # Search all quizzes
            for qid, qdata in self.quizzes.items():
                question = next((q for q in qdata.get("questions", []) if q.get("id") == question_id), None)
                if question:
                    quiz_data = qdata
                    quiz_id = qid
                    break
        
        if not question:
            raise ValueError(f"Question with ID '{question_id}' not found")
        
        if not quiz_data:
            raise ValueError(f"Quiz data not found for question '{question_id}'")
        
        topic = quiz_data.get("topic", "unknown")
        question_type = question.get("type", "multiple_choice")
        correct_answer = question.get("correct_answer", "")
        answer_index = question.get("answer_index", -1)
        options = question.get("options", [])
        explanation = question.get("explanation", "No explanation provided")
        
        # Grade based on question type
        is_correct = False
        score = 0.0
        
        if question_type == "multiple_choice":
            # For multiple choice, check if answer matches index or text
            student_answer_clean = student_answer.strip().upper()
            
            # Check if answer is an option letter (A, B, C, D)
            if student_answer_clean in ['A', 'B', 'C', 'D']:
                student_index = ord(student_answer_clean) - ord('A')
                is_correct = (student_index == answer_index)
            # Check if answer matches the correct answer text
            elif student_answer_clean == correct_answer.upper():
                is_correct = True
            # Check if answer matches any option by index
            elif student_answer_clean.isdigit():
                student_index = int(student_answer_clean) - 1  # Convert 1-based to 0-based
                is_correct = (student_index == answer_index)
            # Check if answer matches option text
            elif options and student_answer_clean in [opt.upper() for opt in options]:
                student_index = [opt.upper() for opt in options].index(student_answer_clean)
                is_correct = (student_index == answer_index)
            else:
                # Use string similarity as fallback
                similarity = SequenceMatcher(None, student_answer_clean, correct_answer.upper()).ratio()
                is_correct = (similarity > 0.8)
                score = similarity
            
            if is_correct:
                score = 1.0
            else:
                score = 0.0
                
        else:  # short_answer
            # For short answer, use string similarity or LLM grading
            correct_answer_clean = correct_answer.lower().strip()
            student_answer_clean = student_answer.lower().strip()
            
            # Calculate similarity
            similarity = SequenceMatcher(None, student_answer_clean, correct_answer_clean).ratio()
            
            # Consider correct if similarity > 0.7
            is_correct = (similarity > 0.7)
            score = similarity
            
            # Optionally use LLM for more nuanced grading
            if not is_correct and similarity > 0.5:
                try:
                    grading_prompt = f"""You are grading a student's answer to a quiz question.

Question: {question.get('question', '')}
Correct Answer: {correct_answer}
Student Answer: {student_answer}

Is the student's answer correct or partially correct? Consider:
- Exact matches are correct
- Paraphrases with same meaning are correct
- Partially correct answers should get partial credit
- Completely wrong answers are incorrect

Respond with JSON only:
{{
    "correct": true or false,
    "score": 0.0 to 1.0,
    "reasoning": "Brief explanation"
}}

Return only valid JSON."""
                    
                    response_text = self._call_gemini(grading_prompt, max_tokens=256, temperature=0.0)
                    grading_result = self._extract_json_from_response(response_text)
                    
                    is_correct = grading_result.get("correct", False)
                    score = grading_result.get("score", similarity)
                    if grading_result.get("reasoning"):
                        explanation = f"{explanation}\n\nGrading note: {grading_result['reasoning']}"
                except Exception:
                    # Fallback to similarity if LLM grading fails
                    pass
        
        # Save incorrect answer to memory if user_id provided
        if not is_correct and user_id:
            wrong_answer_entry = {
                "topic": topic,
                "question_id": question_id,
                "student_answer": student_answer,
                "correct_answer": correct_answer,
                "quiz_id": quiz_id,
                "timestamp": self.memory_store._get_timestamp(),
                "tags": ["wrong_answer", topic]
            }
            
            # Append to user's wrong answers list
            self.memory_store.append_to_list(f"user:{user_id}:wrong_answers", wrong_answer_entry)
        
        return {
            "correct": is_correct,
            "score": score,
            "explanation": explanation,
            "correct_answer": correct_answer,
            "question_id": question_id
        }
    
    def handle_message(self, message: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle incoming messages and route to appropriate action handlers.
        
        Supported actions:
        - generate_quiz: Generate quiz questions
        - grade_answer: Grade a student's answer
        
        Args:
            message: Message dictionary with 'action' and 'payload' keys
            context: Optional context with user info, etc.
            
        Returns:
            Response dictionary with status and payload
        """
        action = message.get("action", "unknown")
        payload = message.get("payload", {})
        request_id = message.get("request_id", "unknown")
        
        try:
            if action == "generate_quiz":
                topic = payload.get("topic", "")
                n_questions = payload.get("n_questions", 5)
                difficulty = payload.get("difficulty", "intermediate")
                
                if not topic:
                    raise ValueError("Topic is required for generate_quiz action")
                
                quiz_data = self.generate_quiz(topic, n_questions, difficulty)
                
                return {
                    "status": "ok",
                    "payload": quiz_data,
                    "request_id": request_id,
                    "meta": {"agent": self.name, "action": action}
                }
                
            elif action == "grade_answer":
                question_id = payload.get("question_id", "")
                student_answer = payload.get("student_answer", "")
                quiz_id = payload.get("quiz_id")
                user_id = payload.get("user_id") or (context.get("user_id") if context else None)
                
                if not question_id or not student_answer:
                    raise ValueError("question_id and student_answer are required for grade_answer action")
                
                grading_result = self.grade_answer(question_id, student_answer, quiz_id, user_id)
                
                return {
                    "status": "ok",
                    "payload": grading_result,
                    "request_id": request_id,
                    "meta": {"agent": self.name, "action": action}
                }
            else:
                return {
                    "status": "error",
                    "payload": {"error": f"Unknown action: {action}"},
                    "request_id": request_id,
                    "meta": {"agent": self.name}
                }
            
        except Exception as e:
            return {
                "status": "error",
                "payload": {"error": str(e)},
                "request_id": request_id,
                "meta": {"agent": self.name, "action": action}
            }


# Example usage
if __name__ == "__main__":
    # Initialize quiz agent
    quiz_agent = QuizAgent(name="quiz_agent")
    
    # Generate a quiz
    print("Generating quiz...")
    quiz = quiz_agent.generate_quiz("linear regression", n_questions=3)
    print(f"✓ Quiz generated: {quiz['quiz_id']}")
    print(f"  Topic: {quiz['topic']}")
    print(f"  Questions: {len(quiz['questions'])}")
    
    # Grade an answer
    if quiz['questions']:
        first_question = quiz['questions'][0]
        question_id = first_question['id']
        correct_answer = first_question.get('correct_answer', '')
        
        print(f"\nGrading answer for question: {first_question['question']}")
        print(f"Correct answer: {correct_answer}")
        
        # Test with correct answer
        result = quiz_agent.grade_answer(
            question_id, 
            correct_answer,
            quiz_id=quiz['quiz_id'],
            user_id="demo_user"
        )
        print(f"\n✓ Grading result:")
        print(f"  Correct: {result['correct']}")
        print(f"  Score: {result['score']}")
        print(f"  Explanation: {result['explanation'][:100]}...")

