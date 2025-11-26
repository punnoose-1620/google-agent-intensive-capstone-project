"""
Tutor Agent - Core LLM-powered educational tutor.

This agent provides explanations, examples, and personalized tutoring
using Google's Gemini API.
"""

import os
import json
import re
from typing import Dict, Any, Optional

# Import base Agent class
from ..agent_framework import Agent

# Try to import Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# Try to import loguru for logging
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False


class TutorAgent(Agent):
    """
    Educational tutor agent that provides explanations, examples, and personalized learning.
    
    Uses Google Gemini API to generate educational content with structured outputs.
    """
    
    def __init__(self, name: str = "tutor_agent", api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """
        Initialize the tutor agent.
        
        Args:
            name: Agent name identifier
            api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
            model_name: Gemini model to use (default: "gemini-pro")
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
    
    def _call_gemini(
    self, 
    prompt: str, 
    max_tokens: int = 8192, 
    temperature: float = 0.0,
    max_retries: int = 3
    ) -> str:
        """
        Call Gemini API with the given prompt.
        
        Best Practice: Use temperature=0.0 for deterministic, reproducible outputs.
        This is especially important for evaluation and grading tasks.
        
        Args:
            prompt: Prompt text to send to Gemini
            max_tokens: Maximum tokens in response (default: 512)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
                        Use 0.0 for evaluation, 0.3-0.7 for creative explanations
            max_retries: Maximum number of retry attempts for transient errors (default: 3)
            
        Returns:
            Response text from Gemini
            
        Raises:
            Exception: If API call fails after all retries
        """
        import time
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                logger.info(f"[{self.name}] Calling Gemini API (attempt {attempt + 1}/{max_retries}) - max_tokens={max_tokens}, temperature={temperature}")
                
                # Generate content with specified parameters
                # Note: Gemini API uses max_output_tokens instead of max_tokens
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                logger.debug(f"[{self.name}] Gemini API response received")
                
                # Check if response was truncated
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    logger.debug(f"[{self.name}] Response finish_reason: {finish_reason}")
                    
                    # Check if response was cut off due to token limit
                    if finish_reason == 'MAX_TOKENS':
                        # Response was truncated - try to get what we have but warn
                        response_text = response.text or ""
                        if not response_text or len(response_text) < 100:
                            # Try to get text from parts if available
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                response_text = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                        
                        logger.warning(f"[{self.name}] Response truncated at {len(response_text)} characters (max_tokens={max_tokens})")
                        # Raise a specific error about truncation
                        raise Exception(
                            f"Response truncated due to max_tokens limit ({max_tokens}). "
                            f"Response was cut off at {len(response_text)} characters. "
                            f"Consider increasing max_tokens or simplifying the prompt. "
                            f"Partial response: {response_text[:200]}..."
                        )
                
                # Get response text
                response_text = response.text
                
                # Check if response_text is None or empty
                if not response_text:
                    # Try to extract from candidates if available
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            response_text = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                    
                    if not response_text:
                        logger.error(f"[{self.name}] Empty response from Gemini API")
                        raise Exception("Empty response from Gemini API")
                
                logger.info(f"[{self.name}] Gemini API response received successfully ({len(response_text)} characters)")
                return response_text
                
            except Exception as e:
                last_exception = e
                error_str = str(e)
                
                # Check if it's a retryable error (500, 503, rate limit, etc.)
                is_retryable = (
                    "500" in error_str or 
                    "503" in error_str or 
                    "429" in error_str or
                    "internal error" in error_str.lower() or
                    "rate limit" in error_str.lower() or
                    "quota" in error_str.lower()
                )
                
                if is_retryable and attempt < max_retries - 1:
                    # Exponential backoff: wait 2^attempt seconds
                    wait_time = 2 ** attempt
                    logger.warning(f"[{self.name}] API error (attempt {attempt + 1}/{max_retries}): {error_str[:100]}")
                    logger.info(f"[{self.name}] Retrying in {wait_time} seconds...")
                    print(f"⚠️  API error (attempt {attempt + 1}/{max_retries}): {error_str[:100]}")
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Not retryable or out of retries
                    logger.error(f"[{self.name}] Gemini API call failed: {error_str}")
                    raise Exception(f"Gemini API call failed: {error_str}")
    
        # If we get here, all retries failed
        logger.error(f"[{self.name}] Gemini API call failed after {max_retries} attempts: {str(last_exception)}")
        raise Exception(f"Gemini API call failed after {max_retries} attempts: {str(last_exception)}")

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response, handling cases where response includes markdown or extra text.
        
        Best Practice: LLMs sometimes wrap JSON in markdown code blocks or add explanatory text.
        This function handles common formats: ... ```, ``` ... ```, or plain JSON.
        Uses json.JSONDecoder.raw_decode() for robust parsing that handles all edge cases automatically.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If JSON cannot be extracted or parsed
        """
        logger.debug(f"[{self.name}] Extracting JSON from response ({len(response_text)} characters)")
        
        # Step 1: Try to extract JSON from markdown code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                parsed_json = json.loads(json_str)
                logger.debug(f"[{self.name}] Successfully parsed JSON from markdown code block")
                return parsed_json
            except json.JSONDecodeError:
                # Markdown extraction might have failed due to non-greedy regex
                # Fall through to raw_decode method
                pass
        
        # Step 2: Find the first '{' character and use raw_decode to parse from there
        # raw_decode automatically handles nested objects, arrays, strings, and all edge cases
        start_idx = response_text.find('{')
        if start_idx == -1:
            # No JSON object found
            logger.error(f"[{self.name}] No JSON object found in response")
            raise ValueError(
                f"No JSON object found in response. Response preview: {response_text[:200]}"
            )
    
        # Use JSONDecoder.raw_decode() to parse JSON starting from start_idx
        # This automatically handles:
        # - Nested objects and arrays
        # - Escaped characters in strings
        # - All JSON edge cases
        # Returns: (parsed_object, end_index)
        decoder = json.JSONDecoder()
        try:
            parsed_json, end_idx = decoder.raw_decode(response_text, idx=start_idx)
            logger.debug(f"[{self.name}] Successfully parsed JSON from response (parsed {end_idx - start_idx} characters)")
            return parsed_json
        except json.JSONDecodeError as e:
            # Check if it's a truncation error (incomplete JSON)
            error_pos = getattr(e, 'pos', None)
            if error_pos and error_pos >= len(response_text) - 10:
                # Error near the end - likely truncated
                logger.error(f"[{self.name}] JSON response appears to be truncated at position {error_pos}")
                raise ValueError(
                    f"JSON response appears to be truncated. "
                    f"Response length: {len(response_text)} characters. "
                    f"Error at position: {error_pos}. "
                    f"Response preview (last 500 chars): {response_text[-500:]}"
                )
            else:
                # JSON syntax error
                logger.error(f"[{self.name}] Failed to parse JSON from response: {e}")
                error_msg = f"Failed to parse JSON from response: {e}\n"
                error_msg += f"Response length: {len(response_text)} characters\n"
                error_msg += f"Error position: {getattr(e, 'pos', 'unknown')}\n"
                error_msg += f"Response preview (first 500 chars): {response_text[:500]}\n"
                if start_idx > 0:
                    error_msg += f"JSON starts at position: {start_idx}\n"
                raise ValueError(error_msg)## Benefits

    def handle_message(self, message: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle incoming messages and route to appropriate action handlers.
        
        Supported actions:
        - explain: Provide explanation of a topic
        - summarize_notes: Summarize learning notes
        - create_example: Create an example problem
        - adapt_to_user: Adapt content to user's level/preferences
        
        Args:
            message: Message dictionary with 'action' and 'payload' keys
            context: Optional context with user info, memory, etc.
            
        Returns:
            Response dictionary with status and payload
        """
        action = message.get("action", "unknown")
        payload = message.get("payload", {})
        request_id = message.get("request_id", "unknown")
        
        logger.info(f"[{self.name}] Handling message - action: {action}, request_id: {request_id}")
        
        try:
            if action == "explain":
                logger.debug(f"[{self.name}] Processing explain action for topic: {payload.get('topic', 'N/A')}")
                result = self._handle_explain(payload, context)
            elif action == "summarize_notes":
                logger.debug(f"[{self.name}] Processing summarize_notes action")
                result = self._handle_summarize_notes(payload, context)
            elif action == "create_example":
                logger.debug(f"[{self.name}] Processing create_example action for topic: {payload.get('topic', 'N/A')}")
                result = self._handle_create_example(payload, context)
            elif action == "adapt_to_user":
                logger.debug(f"[{self.name}] Processing adapt_to_user action")
                result = self._handle_adapt_to_user(payload, context)
            else:
                logger.warning(f"[{self.name}] Unknown action received: {action}")
                return {
                    "status": "error",
                    "payload": {"error": f"Unknown action: {action}"},
                    "request_id": request_id,
                    "meta": {"agent": self.name}
                }
            
            logger.info(f"[{self.name}] Successfully processed action: {action}")
            return {
                "status": "ok",
                "payload": result,
                "request_id": request_id,
                "meta": {"agent": self.name, "action": action}
            }
            
        except Exception as e:
            logger.error(f"[{self.name}] Error processing action {action}: {str(e)}")
            return {
                "status": "error",
                "payload": {"error": str(e)},
                "request_id": request_id,
                "meta": {"agent": self.name, "action": action}
            }
    
    def _handle_explain(self, payload: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle explain action - provide explanation of a topic.
        
        Expected payload:
        - topic: str - Topic to explain
        - level: str - Difficulty level (beginner/intermediate/advanced)
        - additional_context: Optional[str] - Additional context or requirements
        """
        topic = payload.get("topic", "")
        level = payload.get("level", "intermediate")
        additional_context = payload.get("additional_context", "")
        
        if not topic:
            logger.error(f"[{self.name}] Explain action called without topic")
            raise ValueError("Topic is required for explain action")
        
        logger.info(f"[{self.name}] Generating explanation for topic: '{topic}' at {level} level")
        
        # Build prompt using structured template
        # Best Practice: Use clear role, instruction, and content structure
        # This helps the model understand the task and produce consistent outputs
        prompt = f"""You are an expert university tutor. Your role is to provide clear, structured explanations.

Role: tutor
Instruction: Explain the topic "{topic}" at {level} level. Provide a structured explanation with summary, step-by-step breakdown, examples, and potential difficulties.

Content: {additional_context if additional_context else "No additional context provided."}

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
Return only valid JSON."""
        
        # Use temperature=0.0 for deterministic, reproducible explanations
        # Best Practice: For educational content, deterministic outputs ensure consistency
        logger.debug(f"[{self.name}] Calling Gemini for explanation (max_tokens=8192)")
        response_text = self._call_gemini(prompt, max_tokens=8192, temperature=0.0)
        result = self._extract_json_from_response(response_text)
        
        logger.info(f"[{self.name}] Explanation generated successfully - summary length: {len(result.get('summary', ''))}")
        
        # Ensure all expected keys are present
        return {
            "summary": result.get("summary", ""),
            "step_by_step": result.get("step_by_step", []),
            "examples": result.get("examples", []),
            "key_equations": result.get("key_equations", []),
            "difficulties": result.get("difficulties", []),
            "further_reading": result.get("further_reading", []),
            "confidence": result.get("confidence", "medium")
        }
    
    def _handle_summarize_notes(self, payload: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle summarize_notes action - summarize learning notes.
        
        Expected payload:
        - notes: str or list - Notes to summarize
        - focus_areas: Optional[list] - Specific areas to focus on
        """
        notes = payload.get("notes", "")
        focus_areas = payload.get("focus_areas", [])
        
        if not notes:
            raise ValueError("Notes are required for summarize_notes action")
        
        # Convert list to string if needed
        if isinstance(notes, list):
            notes = "\n".join([str(note) for note in notes])
        
        focus_text = f"\nFocus on these areas: {', '.join(focus_areas)}" if focus_areas else ""
        
        prompt = f"""You are an expert tutor helping to summarize learning notes.

Role: tutor
Instruction: Summarize the following learning notes into key concepts, important points, and action items.{focus_text}

Content:
{notes}

Return your response as valid JSON only:
{{
    "summary": "Brief overall summary",
    "key_concepts": ["Concept 1", "Concept 2", ...],
    "important_points": ["Point 1", "Point 2", ...],
    "action_items": ["Action 1", "Action 2", ...],
    "gaps": ["Knowledge gap 1", "Knowledge gap 2", ...]
}}

Return only valid JSON."""
        
        response_text = self._call_gemini(prompt, max_tokens=8192, temperature=0.0)
        return self._extract_json_from_response(response_text)
    
    def _handle_create_example(self, payload: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle create_example action - create an example problem.
        
        Expected payload:
        - topic: str - Topic for the example
        - difficulty: str - Difficulty level
        - example_type: Optional[str] - Type of example (problem, demonstration, etc.)
        """
        topic = payload.get("topic", "")
        difficulty = payload.get("difficulty", "intermediate")
        example_type = payload.get("example_type", "problem")
        
        if not topic:
            raise ValueError("Topic is required for create_example action")
        
        prompt = f"""You are an expert tutor creating educational examples.

Role: tutor
Instruction: Create a {example_type} example for the topic "{topic}" at {difficulty} difficulty level.

Content: Create a clear, educational example that demonstrates key concepts.

Return your response as valid JSON only:
{{
    "title": "Example title",
    "problem": "Problem statement or question",
    "solution": "Step-by-step solution",
    "explanation": "Why this solution works",
    "key_concepts": ["Concept 1", "Concept 2", ...],
    "variations": ["Variation 1", "Variation 2", ...]
}}

Return only valid JSON."""
        
        # Use slightly higher temperature for creative examples (0.3)
        # Best Practice: Balance creativity (higher temp) with consistency (lower temp)
        response_text = self._call_gemini(prompt, max_tokens=8192, temperature=0.3)
        return self._extract_json_from_response(response_text)
    
    def _handle_adapt_to_user(self, payload: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle adapt_to_user action - adapt content to user's level and preferences.
        
        Expected payload:
        - content: str - Content to adapt
        - user_level: str - User's current level
        - user_preferences: Optional[dict] - User preferences (learning style, etc.)
        - user_history: Optional[list] - Previous topics or difficulties
        """
        content = payload.get("content", "")
        user_level = payload.get("user_level", "intermediate")
        user_preferences = payload.get("user_preferences", {})
        user_history = payload.get("user_history", [])
        
        if not content:
            raise ValueError("Content is required for adapt_to_user action")
        
        preferences_text = json.dumps(user_preferences) if user_preferences else "None"
        history_text = ", ".join(user_history) if user_history else "None"
        
        prompt = f"""You are an expert tutor adapting content to a specific learner.

Role: tutor
Instruction: Adapt the following content to match the user's level ({user_level}) and preferences.

Content to adapt:
{content}

User preferences: {preferences_text}
User learning history: {history_text}

Return your response as valid JSON only:
{{
    "adapted_content": "Content adapted to user's level and preferences",
    "simplifications": ["Simplification 1", "Simplification 2", ...],
    "enhancements": ["Enhancement 1", "Enhancement 2", ...],
    "personalized_examples": ["Example 1", "Example 2", ...],
    "recommended_next_steps": ["Step 1", "Step 2", ...]
}}

Return only valid JSON."""
        
        response_text = self._call_gemini(prompt, max_tokens=2048, temperature=0.0)
        return self._extract_json_from_response(response_text)


# Example usage
if __name__ == "__main__":
    # Initialize tutor agent
    tutor = TutorAgent(name="tutor_agent")
    
    # Test explain action
    message = {
        "action": "explain",
        "payload": {
            "topic": "bias-variance tradeoff",
            "level": "intermediate"
        },
        "request_id": "test-001"
    }
    
    response = tutor.handle_message(message)
    print("Explain response:")
    print(json.dumps(response, indent=2))

