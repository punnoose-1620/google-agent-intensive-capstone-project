"""
Evaluator Agent - LLM-as-judge for evaluating educational content.

This agent evaluates explanations, summaries, and quizzes using Google's Gemini API
with deterministic settings for consistent scoring.
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
from collections import Counter

# Import base Agent class
from ..agent_framework import Agent

# Import prompt template
from .prompts import EVALUATOR_PROMPT

# Try to import Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


class EvaluatorAgent(Agent):
    """
    Evaluator agent that assesses the quality of educational content.
    
    Uses Google Gemini API with temperature=0 for deterministic, reproducible evaluations.
    Includes fallback heuristics for when API is unavailable.
    """
    
    def __init__(self, name: str = "evaluator_agent", api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """
        Initialize the evaluator agent.
        
        Args:
            name: Agent name identifier
            api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
            model_name: Gemini model to use (default: "gemini-pro")
        """
        super().__init__(name=name)
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.use_gemini = GEMINI_AVAILABLE and self.api_key is not None
        
        if self.use_gemini:
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
        else:
            self.model = None
            if not GEMINI_AVAILABLE:
                print("Warning: google-generativeai not available. Using fallback heuristics only.")
            elif not self.api_key:
                print("Warning: GEMINI_API_KEY not found. Using fallback heuristics only.")
    
    def _call_gemini(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.0
    ) -> str:
        """
        Call Gemini API with the given prompt.
        
        Uses temperature=0.0 for deterministic, reproducible evaluations.
        
        Args:
            prompt: Prompt text to send to Gemini
            max_tokens: Maximum tokens in response (default: 512)
            temperature: Sampling temperature (always 0.0 for evaluations)
            
        Returns:
            Response text from Gemini
            
        Raises:
            Exception: If API call fails
        """
        if not self.use_gemini:
            raise Exception("Gemini API not available")
        
        try:
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,  # Always 0.0 for deterministic evaluations
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
    
    def _extract_keyphrases(self, text: str, max_phrases: int = 20) -> List[str]:
        """
        Extract keyphrases from text for fallback heuristic.
        
        Args:
            text: Text to extract keyphrases from
            max_phrases: Maximum number of keyphrases to extract
            
        Returns:
            List of keyphrases (lowercased, cleaned)
        """
        # Simple keyphrase extraction: words and phrases
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        # Extract words (2+ characters, alphanumeric)
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        
        # Filter stop words and get unique words
        keyphrases = [w for w in words if w not in stop_words]
        
        # Count frequency and get top phrases
        word_counts = Counter(keyphrases)
        top_phrases = [phrase for phrase, count in word_counts.most_common(max_phrases)]
        
        # Also extract 2-3 word phrases
        phrases = re.findall(r'\b[a-zA-Z]{2,}\s+[a-zA-Z]{2,}(?:\s+[a-zA-Z]{2,})?', text.lower())
        phrases = [p for p in phrases if not any(sw in p for sw in stop_words)]
        phrase_counts = Counter(phrases)
        top_phrases.extend([p for p, count in phrase_counts.most_common(max_phrases // 2)])
        
        return list(set(top_phrases[:max_phrases]))
    
    def _calculate_string_overlap_score(self, source_text: str, candidate_text: str) -> Dict[str, float]:
        """
        Calculate string overlap scores between source and candidate text.
        
        Fallback heuristic when Gemini API is unavailable.
        Uses keyphrase overlap to estimate accuracy and relevance.
        
        Args:
            source_text: Source/reference text
            candidate_text: Candidate text to evaluate
            
        Returns:
            Dictionary with overlap scores (0.0 to 1.0)
        """
        # Extract keyphrases from both texts
        source_phrases = set(self._extract_keyphrases(source_text))
        candidate_phrases = set(self._extract_keyphrases(candidate_text))
        
        if not source_phrases:
            return {
                "accuracy": 0.5,
                "clarity": 0.5,
                "usefulness": 0.5,
                "overlap_score": 0.0
            }
        
        # Calculate overlap metrics
        common_phrases = source_phrases.intersection(candidate_phrases)
        overlap_ratio = len(common_phrases) / len(source_phrases) if source_phrases else 0.0
        coverage_ratio = len(common_phrases) / len(candidate_phrases) if candidate_phrases else 0.0
        
        # Map overlap to scores (0-5 scale, then normalize to 0-1)
        # Higher overlap = higher accuracy
        accuracy_score = min(overlap_ratio * 5, 5.0) / 5.0
        
        # Clarity: based on candidate text length and structure
        # Shorter, focused text tends to be clearer
        word_count = len(candidate_text.split())
        clarity_score = 0.7 if 50 <= word_count <= 300 else 0.5
        
        # Usefulness: combination of overlap and coverage
        usefulness_score = (overlap_ratio * 0.6 + coverage_ratio * 0.4)
        usefulness_score = min(usefulness_score * 5, 5.0) / 5.0
        
        return {
            "accuracy": accuracy_score,
            "clarity": clarity_score,
            "usefulness": usefulness_score,
            "overlap_score": overlap_ratio,
            "common_phrases_count": len(common_phrases),
            "source_phrases_count": len(source_phrases),
            "candidate_phrases_count": len(candidate_phrases)
        }
    
    def handle_message(self, message: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle incoming messages and route to appropriate action handlers.
        
        Supported actions:
        - evaluate_summary: Evaluate a summary against source text
        - evaluate_quiz: Evaluate quiz questions for quality
        
        Args:
            message: Message dictionary with 'action' and 'payload' keys
            context: Optional context with additional information
            
        Returns:
            Response dictionary with status and payload
        """
        action = message.get("action", "unknown")
        payload = message.get("payload", {})
        request_id = message.get("request_id", "unknown")
        
        try:
            if action == "evaluate_summary":
                result = self._handle_evaluate_summary(payload, context)
            elif action == "evaluate_quiz":
                result = self._handle_evaluate_quiz(payload, context)
            else:
                return {
                    "status": "error",
                    "payload": {"error": f"Unknown action: {action}"},
                    "request_id": request_id,
                    "meta": {"agent": self.name}
                }
            
            return {
                "status": "ok",
                "payload": result,
                "request_id": request_id,
                "meta": {"agent": self.name, "action": action}
            }
            
        except Exception as e:
            return {
                "status": "error",
                "payload": {"error": str(e)},
                "request_id": request_id,
                "meta": {"agent": self.name, "action": action}
            }
    
    def _handle_evaluate_summary(self, payload: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle evaluate_summary action.
        
        Expected payload:
        - source_text: str - Source/reference text
        - candidate: str - Candidate summary to evaluate
        - evaluation_criteria: Optional[str] - Specific criteria to focus on
        """
        source_text = payload.get("source_text", "")
        candidate = payload.get("candidate", "")
        evaluation_criteria = payload.get("evaluation_criteria")
        
        if not source_text or not candidate:
            raise ValueError("Both source_text and candidate are required for evaluation")
        
        # Try to use Gemini API first
        if self.use_gemini:
            try:
                prompt = EVALUATOR_PROMPT(source_text, candidate, evaluation_criteria)
                response_text = self._call_gemini(prompt, max_tokens=512, temperature=0.0)
                result = self._extract_json_from_response(response_text)
                
                # Ensure required fields are present
                return {
                    "accuracy": result.get("accuracy_score", 0) / 5.0 if isinstance(result.get("accuracy_score"), (int, float)) else 0.5,
                    "clarity": result.get("clarity_score", 0) / 5.0 if isinstance(result.get("clarity_score"), (int, float)) else 0.5,
                    "usefulness": result.get("usefulness_score", 0) / 5.0 if isinstance(result.get("usefulness_score"), (int, float)) else 0.5,
                    "hallucinations": result.get("hallucinated_claims", []),
                    "method": "gemini",
                    "raw_scores": {
                        "accuracy_score": result.get("accuracy_score", 0),
                        "clarity_score": result.get("clarity_score", 0),
                        "usefulness_score": result.get("usefulness_score", 0),
                        "overall_score": result.get("overall_score", 0)
                    }
                }
            except Exception as e:
                print(f"Warning: Gemini evaluation failed: {e}. Using fallback heuristic.")
                # Fall through to fallback
        
        # Fallback: use string overlap heuristic
        overlap_scores = self._calculate_string_overlap_score(source_text, candidate)
        return {
            "accuracy": overlap_scores["accuracy"],
            "clarity": overlap_scores["clarity"],
            "usefulness": overlap_scores["usefulness"],
            "hallucinations": [],  # Cannot detect hallucinations with heuristic
            "method": "heuristic",
            "overlap_details": overlap_scores
        }
    
    def _handle_evaluate_quiz(self, payload: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle evaluate_quiz action.
        
        Expected payload:
        - quiz_data: dict - Quiz data with questions
        - source_topic: Optional[str] - Source topic for reference
        - evaluation_criteria: Optional[str] - Specific criteria
        """
        quiz_data = payload.get("quiz_data", {})
        source_topic = payload.get("source_topic", "")
        evaluation_criteria = payload.get("evaluation_criteria")
        
        if not quiz_data:
            raise ValueError("quiz_data is required for quiz evaluation")
        
        questions = quiz_data.get("questions", [])
        if not questions:
            raise ValueError("Quiz must contain at least one question")
        
        # Build candidate text from quiz
        candidate_text = f"Quiz on topic: {source_topic or 'unknown'}\n\n"
        for q in questions:
            candidate_text += f"Q: {q.get('question', '')}\n"
            if 'options' in q:
                candidate_text += f"Options: {', '.join(q.get('options', []))}\n"
            candidate_text += f"Answer: {q.get('correct_answer', '')}\n\n"
        
        # For quiz evaluation, we evaluate against expected quiz quality
        # Create a reference text about good quiz design
        source_text = f"""A good quiz on {source_topic or 'a topic'} should:
- Test understanding of key concepts
- Have clear, unambiguous questions
- Include appropriate difficulty level
- Provide correct answers with explanations
- Cover important aspects of the topic"""
        
        # Try to use Gemini API first
        if self.use_gemini:
            try:
                prompt = EVALUATOR_PROMPT(
                    source_text, 
                    candidate_text, 
                    evaluation_criteria or "quiz quality, question clarity, answer accuracy"
                )
                response_text = self._call_gemini(prompt, max_tokens=512, temperature=0.0)
                result = self._extract_json_from_response(response_text)
                
                return {
                    "accuracy": result.get("accuracy_score", 0) / 5.0 if isinstance(result.get("accuracy_score"), (int, float)) else 0.5,
                    "clarity": result.get("clarity_score", 0) / 5.0 if isinstance(result.get("clarity_score"), (int, float)) else 0.5,
                    "usefulness": result.get("usefulness_score", 0) / 5.0 if isinstance(result.get("usefulness_score"), (int, float)) else 0.5,
                    "hallucinations": result.get("hallucinated_claims", []),
                    "method": "gemini",
                    "raw_scores": {
                        "accuracy_score": result.get("accuracy_score", 0),
                        "clarity_score": result.get("clarity_score", 0),
                        "usefulness_score": result.get("usefulness_score", 0),
                        "overall_score": result.get("overall_score", 0)
                    },
                    "quiz_metrics": {
                        "total_questions": len(questions),
                        "questions_with_explanations": sum(1 for q in questions if q.get('explanation')),
                        "questions_with_options": sum(1 for q in questions if 'options' in q)
                    }
                }
            except Exception as e:
                print(f"Warning: Gemini evaluation failed: {e}. Using fallback heuristic.")
                # Fall through to fallback
        
        # Fallback: use string overlap heuristic
        overlap_scores = self._calculate_string_overlap_score(source_text, candidate_text)
        return {
            "accuracy": overlap_scores["accuracy"],
            "clarity": overlap_scores["clarity"],
            "usefulness": overlap_scores["usefulness"],
            "hallucinations": [],
            "method": "heuristic",
            "overlap_details": overlap_scores,
            "quiz_metrics": {
                "total_questions": len(questions),
                "questions_with_explanations": sum(1 for q in questions if q.get('explanation')),
                "questions_with_options": sum(1 for q in questions if 'options' in q)
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize evaluator agent
    evaluator = EvaluatorAgent(name="evaluator_agent")
    
    # Test evaluate_summary
    source = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    candidate = "Machine learning is part of AI and allows systems to learn from data."
    
    message = {
        "action": "evaluate_summary",
        "payload": {
            "source_text": source,
            "candidate": candidate
        },
        "request_id": "test-001"
    }
    
    response = evaluator.handle_message(message)
    print("Evaluation response:")
    print(json.dumps(response, indent=2))

