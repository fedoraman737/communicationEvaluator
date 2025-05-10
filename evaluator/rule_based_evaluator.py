import re
from datetime import datetime
import jellyfish  # For fuzzy string matching
import language_tool_python  # For grammar checking
from models.response import Response
from models.evaluation import Evaluation

# Attempt to import lexicons
try:
    from .lexicons.acknowledgement_phrases import ACKNOWLEDGEMENT_PHRASES
    from .lexicons.blunt_negative_phrases import BLUNT_NEGATIVE_PHRASES, CONTEXTUAL_NEGATIVES
    from .lexicons.solution_phrases import POSITIVE_FRAMING_PHRASES, SOLUTION_ORIENTED_KEYWORDS
    from .lexicons.empathy_expressions import EMPATHY_EXPRESSIONS, EMPATHY_KEYWORDS
    from .lexicons.uncertainty_phrases import UNCERTAINTY_PHRASES, LESS_CONFIDENT_PHRASES, FLEXIBILITY_PHRASES, APPROPRIATE_CAUTION_PHRASES
    from .lexicons.courtesy_phrases import COURTESY_PHRASES, POLITE_INTERJECTIONS
except ImportError:
    # Fallback for environments where lexicons might not be in a package
    # This might happen during testing or if the structure is different.
    # For robust solution, ensure PYTHONPATH or package structure is correct.
    print("Warning: Lexicons not found in expected package structure. Using empty lists.")
    ACKNOWLEDGEMENT_PHRASES = []
    BLUNT_NEGATIVE_PHRASES = []
    CONTEXTUAL_NEGATIVES = []
    POSITIVE_FRAMING_PHRASES = {}
    SOLUTION_ORIENTED_KEYWORDS = []
    EMPATHY_EXPRESSIONS = {}
    EMPATHY_KEYWORDS = []
    UNCERTAINTY_PHRASES = []
    LESS_CONFIDENT_PHRASES = []
    FLEXIBILITY_PHRASES = []
    APPROPRIATE_CAUTION_PHRASES = []
    COURTESY_PHRASES = []
    POLITE_INTERJECTIONS = []


class RuleBasedEvaluator:
    """Evaluates advisor responses based on a set of predefined rules and patterns."""

    def __init__(self):
        # Initialize language tool for grammar checking
        try:
            self.language_tool = language_tool_python.LanguageTool('en-US')
        except:
            # Fallback if language tool initialization fails
            print("Warning: LanguageTool initialization failed. Grammar checking will be limited.")
            self.language_tool = None
            
        # Fuzzy matching threshold
        self.fuzzy_match_threshold = 0.85  # Levenshtein distance ratio threshold
        
        # Define minimum acceptable response properties
        self.min_word_count = 10  # Minimum acceptable word count
        self.min_sentence_count = 1  # Minimum number of complete sentences

    def _fuzzy_string_match(self, text: str, target: str) -> bool:
        """Uses fuzzy string matching to handle minor spelling errors."""
        # Exact match first (faster)
        if target.lower() in text.lower():
            return True
            
        # Try fuzzy matching for longer phrases (to avoid false positives on short words)
        if len(target) > 4:
            # Check each potential match position
            text_words = text.lower().split()
            target_words = target.lower().split()
            
            # For single word targets
            if len(target_words) == 1:
                for word in text_words:
                    if jellyfish.jaro_winkler_similarity(word, target) > self.fuzzy_match_threshold:
                        return True
            
            # For multi-word targets
            else:
                # Sliding window approach for multi-word matching
                for i in range(len(text_words) - len(target_words) + 1):
                    text_segment = " ".join(text_words[i:i+len(target_words)])
                    if jellyfish.jaro_winkler_similarity(text_segment, target.lower()) > self.fuzzy_match_threshold:
                        return True
                        
        return False

    def _check_phrases(self, text_to_analyze: str, phrases: list[str] | dict, is_dict_with_explanations: bool = False) -> tuple[bool, list[str]]:
        """Helper to check for presence of phrases and return found phrases/explanations."""
        found_items = []
        text_lower = text_to_analyze.lower() # Ensure consistent case for matching
        
        source_list = phrases.keys() if is_dict_with_explanations else phrases

        for phrase in source_list:
            # Basic check: phrase in text
            if self._fuzzy_string_match(text_lower, phrase):
                if is_dict_with_explanations:
                    found_items.append(f"{phrase} ({phrases[phrase]})") # Add phrase and its explanation
                else:
                    found_items.append(phrase)
            # More sophisticated matching using regex for certain patterns
            elif "[" in phrase and "]" in phrase:
                # Handle templated phrases like "while [negative outcome], we can [positive action]"
                pattern = phrase.replace("[", r"(?:\w+\s*)+").replace("]", "")
                pattern = r'\b' + re.escape(pattern.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    if is_dict_with_explanations:
                        found_items.append(f"{phrase} ({phrases[phrase]})")
                    else:
                        found_items.append(phrase)

        return len(found_items) > 0, found_items

    def _evaluate_response_coherence(self, text: str) -> tuple[float, list[str]]:
        """Evaluates grammar, coherence and informativeness of the response."""
        feedback = []
        score = 5.0  # Start with neutral score
        
        # 1. Check response length (word count)
        words = text.split()
        word_count = len(words)
        
        if word_count < 5:
            score = 1.0  # Severely penalize extremely short responses
            feedback.append(f"Response is too short ({word_count} words). Professional communication typically requires more detailed explanations.")
        elif word_count < self.min_word_count:
            score = 3.0  # Penalize short responses
            feedback.append(f"Response is quite brief ({word_count} words). Consider providing more detailed information.")
        
        # 2. Check sentence structure
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 2]  # Basic check for "complete" sentences
        
        if len(complete_sentences) < self.min_sentence_count:
            score -= 2.0
            feedback.append("Response lacks complete sentences. Professional communication should use proper sentence structure.")
        
        # 3. Check for grammar using LanguageTool if available
        if self.language_tool:
            try:
                matches = self.language_tool.check(text)
                grammar_issues = len(matches)
                
                if grammar_issues > 5:
                    score -= 2.0
                    feedback.append(f"Response contains multiple grammatical errors ({grammar_issues}). Professional communication should be grammatically correct.")
                elif grammar_issues > 2:
                    score -= 1.0
                    feedback.append("Response contains some grammatical errors. Review for clarity and correctness.")
                elif grammar_issues == 0 and word_count > 20:
                    # Bonus for grammatically correct longer responses
                    score += 0.5
                    feedback.append("Response is grammatically well-formed.")
            except:
                # Fallback if grammar checking fails
                pass
        
        # 4. Check for information content (unique content words)
        # Filter out stop words and count unique content words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'that', 'this', 'these', 'those', 
                     'it', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
                     'does', 'did', 'will', 'would', 'should', 'can', 'could', 'to', 'for', 'with', 'in', 
                     'on', 'at', 'from', 'by', 'about', 'as', 'of'}
        content_words = set(word.lower() for word in words if word.lower() not in stop_words and len(word) > 2)
        
        if len(content_words) < 5 and word_count > 10:
            score -= 1.0
            feedback.append("Response lacks informational content. Use more specific, varied vocabulary relevant to the customer's needs.")
        elif len(content_words) > 15:
            score += 0.5
            feedback.append("Response contains a good variety of informational content.")
        
        # 5. Check for repetition (indicator of low quality)
        word_frequencies = {}
        for word in words:
            if len(word) > 3:  # Only count substantial words
                word_frequencies[word.lower()] = word_frequencies.get(word.lower(), 0) + 1
        
        repeat_words = [word for word, count in word_frequencies.items() 
                        if count > 3 and word.lower() not in stop_words]
        
        if repeat_words:
            score -= 0.5
            feedback.append(f"Response contains excessive repetition of words: {', '.join(repeat_words[:3])}.")
            
        return max(0.0, min(10.0, score)), feedback

    def _check_context_relevance(self, response_text: str, query_text: str = None) -> tuple[float, list[str]]:
        """Analyzes if the response addresses the specific context of the query."""
        # Skip if no query text is provided
        if not query_text:
            return 5.0, ["No query context provided to evaluate relevance"]
        
        feedback = []
        score = 5.0
        
        # Extract key terms from query (simplistic approach, could be enhanced with NLP)
        query_words = set(re.findall(r'\b\w{3,}\b', query_text.lower()))
        response_words = set(re.findall(r'\b\w{3,}\b', response_text.lower()))
        
        # Check overlap of key terms
        overlap = query_words.intersection(response_words)
        overlap_ratio = len(overlap) / len(query_words) if query_words else 0
        
        if overlap_ratio > 0.6:
            score += 2.5
            feedback.append(f"Good contextual relevance: Response addresses {len(overlap)} key topics from the query.")
        elif overlap_ratio > 0.3:
            score += 1.0
            feedback.append("Moderate contextual relevance: Response addresses some key topics but could be more specific.")
        else:
            score -= 1.5
            feedback.append("Low contextual relevance: Response doesn't address many key topics from the query.")
            
        # Look for specific question answering patterns
        if "?" in query_text:
            # Find direct answers to questions
            if any(phrase in response_text.lower() for phrase in ["yes", "no", "it is", "it isn't", "it is not", "that's correct", "that is correct"]):
                score += 1.0
                feedback.append("Response directly answers questions posed in the query.")
            else:
                feedback.append("Consider directly answering questions posed in the query.")
                
        return min(10.0, max(0.0, score)), feedback

    def _evaluate_personalization(self, text_to_analyze: str, customer_info: dict = None) -> tuple[float, list[str]]:
        """Evaluates how well the response is personalized to the customer."""
        score = 5.0  # Start with neutral score
        strengths = []
        areas_for_improvement = []
        
        # If no customer info is provided, we can only assess generic personalization
        if not customer_info:
            # Check for generic personalization indicators
            if re.search(r'\byour\b', text_to_analyze.lower()):
                score += 1.0
                strengths.append("Uses 'your' to personalize the message")
            else:
                areas_for_improvement.append("Consider using more personalized language like 'your' when referring to customer's situation")
                
            return score, strengths + areas_for_improvement
            
        # With customer info, we can check for specific personalization
        customer_name = customer_info.get('name', '')
        if customer_name and customer_name.lower() in text_to_analyze.lower():
            score += 2.0
            strengths.append(f"Uses customer's name ('{customer_name}')")
        else:
            areas_for_improvement.append("Consider addressing the customer by name")
            
        # Check if response mentions specific details from customer account/history
        for key, value in customer_info.items():
            if key != 'name' and str(value).lower() in text_to_analyze.lower():
                score += 1.5
                strengths.append(f"References specific customer information ({key})")
                
        # Check for conversation continuity if previous interactions exist
        if 'previous_interaction' in customer_info:
            prev_text = customer_info['previous_interaction'].lower()
            current_text = text_to_analyze.lower()
            
            # Check for references to previous conversation
            if "as we discussed" in current_text or "as mentioned earlier" in current_text:
                score += 1.0
                strengths.append("Maintains conversation continuity with reference to previous interactions")
            elif any(phrase in prev_text and phrase in current_text for phrase in SOLUTION_ORIENTED_KEYWORDS):
                score += 0.5
                strengths.append("Continues discussion of solutions from previous interaction")
                
        return min(10.0, max(0.0, score)), strengths + areas_for_improvement

    def _evaluate_empathy_and_rapport(self, text_to_analyze: str, strengths: list[str], areas_for_improvement: list[str], feedback_log: list[str]) -> float:
        """Evaluates empathy and rapport building."""
        score = 5.0  # Start with a neutral score
        
        # 1. Basic Acknowledgement (using ACKNOWLEDGEMENT_PHRASES)
        found_ack, ack_phrases = self._check_phrases(text_to_analyze, ACKNOWLEDGEMENT_PHRASES)
        if found_ack:
            score += 2.0
            strengths.append(f"Good acknowledgement: Used phrases like '{ack_phrases[0]}'.")
            feedback_log.append("Demonstrated acknowledgement.")
        else:
            score -= 2.0
            areas_for_improvement.append("Consider explicitly acknowledging the customer's statements or feelings (e.g., 'I understand', 'I see how that's frustrating').")
            feedback_log.append("Lacked explicit acknowledgement.")

        # 2. Deeper Empathy Expressions (using EMPATHY_EXPRESSIONS)
        found_empathy, empathy_expr = self._check_phrases(text_to_analyze, EMPATHY_EXPRESSIONS, is_dict_with_explanations=True)
        if found_empathy:
            score += 2.5
            for expr in empathy_expr[:2]:  # Limit to first two for conciseness
                strengths.append(f"Showed good empathy: {expr}")
            feedback_log.append("Used empathetic language effectively.")
        else:
            # Check for empathy keywords even if no complete expressions matched
            found_keywords, keywords = self._check_phrases(text_to_analyze, EMPATHY_KEYWORDS)
            if found_keywords:
                score += 1.0  # Smaller bonus for just keywords
                strengths.append(f"Used empathetic keywords like '{keywords[0]}'.")
                feedback_log.append("Used some empathetic keywords.")
            else:
                # Not finding specific deep empathy phrases might not always be a major negative if acknowledgement is present
                # but it's an area for improvement if the situation calls for it.
                areas_for_improvement.append("Consider using more specific empathetic phrases to validate the customer's feelings if the situation is sensitive (e.g., 'I can see that's frustrating because...').")
                feedback_log.append("Could enhance empathetic expressions.")
        
        # 3. Check for personalized empathy (beyond stock phrases)
        if re.search(r'I understand .+because .+', text_to_analyze, re.IGNORECASE):
            score += 1.0  # Bonus for personalized explanations of empathy
            strengths.append("Used personalized empathy with specific reasoning.")
            feedback_log.append("Demonstrated detailed understanding with personalized empathy.")
        
        return round(max(0.0, min(10.0, score)), 1)

    def _evaluate_positioning_and_solutions(self, text_to_analyze: str, strengths: list[str], areas_for_improvement: list[str], feedback_log: list[str]) -> float:
        """Evaluates how news is positioned and if solutions are offered."""
        score = 5.0

        # 1. Check for Blunt Negative Positioning
        found_blunt_negative, neg_phrases = self._check_phrases(text_to_analyze, BLUNT_NEGATIVE_PHRASES)
        
        # Check for contextual negatives which are more acceptable in certain contexts
        found_contextual_negative, contextual_phrases = self._check_phrases(text_to_analyze, CONTEXTUAL_NEGATIVES)
        
        # Check if negative phrasing is followed by solution-oriented language
        has_mitigating_solutions = False
        if found_blunt_negative or found_contextual_negative:
            phrases_to_check = neg_phrases if found_blunt_negative else contextual_phrases
            
            # Look for solution phrases within 150 characters after the negative phrase
            for neg_phrase in phrases_to_check:
                neg_idx = text_to_analyze.lower().find(neg_phrase.lower())
                if neg_idx >= 0:
                    context_after = text_to_analyze[neg_idx:neg_idx+150].lower()
                    for solution_keyword in SOLUTION_ORIENTED_KEYWORDS:
                        if solution_keyword.lower() in context_after:
                            has_mitigating_solutions = True
                            break
            
            # Handling blunt negative phrases more severely
            if found_blunt_negative:
                if has_mitigating_solutions:
                    # Less severe penalty if solutions are offered immediately after negative news
                    score -= 1.5
                    areas_for_improvement.append(f"Negative language detected: '{neg_phrases[0]}', though solutions were offered afterward. Consider leading with what's possible rather than what's not.")
                    feedback_log.append("Used negative phrasing but followed with solutions.")
                else:
                    score = 2.0  # Severe penalty only if no solutions follow the negative language
                    areas_for_improvement.append(f"Avoided blunt negative language. Found: '{neg_phrases[0]}'. Focus on what is possible or alternative solutions.")
                    feedback_log.append("Used blunt negative phrasing without solutions.")
            # Handling contextual negatives more leniently
            elif found_contextual_negative:
                has_policy_explanation = "policy" in text_to_analyze.lower() or "guidelines" in text_to_analyze.lower() or "criteria" in text_to_analyze.lower()
                
                if has_mitigating_solutions and has_policy_explanation:
                    # No penalty if properly explained and alternative solutions offered
                    strengths.append("Appropriately explained policy limitations while offering alternatives.")
                    feedback_log.append("Used necessary negative language with proper context and solutions.")
                elif has_mitigating_solutions:
                    # Very small penalty if solutions offered but explanation lacking
                    score -= 0.5
                    areas_for_improvement.append("When explaining limitations, provide more specific policy details while maintaining the good solution focus.")
                    feedback_log.append("Used contextual negative with solutions but limited explanation.")
                elif has_policy_explanation:
                    # Medium penalty if explained but no solutions
                    score -= 1.0
                    areas_for_improvement.append("Policy explanation is good, but also suggest alternatives or next steps when delivering negative news.")
                    feedback_log.append("Used contextual negative with explanation but no solutions.")
                else:
                    # Larger penalty if neither explained nor solutions offered
                    score -= 2.0
                    areas_for_improvement.append(f"Phrase '{contextual_phrases[0]}' should be accompanied by policy explanation and alternative solutions.")
                    feedback_log.append("Used contextual negative without proper handling.")
        else:
            strengths.append("Avoided direct negative language when delivering information.")
            feedback_log.append("Successfully avoided blunt negatives.")
            score += 1.0 # Reward for not being blunt

        # 2. Reward Positive Framing and Solution Orientation
        found_positive_framing, pos_frames = self._check_phrases(text_to_analyze, POSITIVE_FRAMING_PHRASES, is_dict_with_explanations=True)
        found_solution_keywords, sol_keys = self._check_phrases(text_to_analyze, SOLUTION_ORIENTED_KEYWORDS)

        if found_positive_framing:
            score += 2.5
            for frame in pos_frames[:2]:  # Limit to first two for conciseness
                strengths.append(f"Used positive framing: {frame}")
            feedback_log.append("Effectively used positive framing.")
        elif found_solution_keywords: # If no specific framing phrase, but solution keywords are there
            score += 1.5  # Increased bonus for having solution keywords
            strengths.append(f"Oriented towards solutions with keywords like '{sol_keys[0]}'.")
            feedback_log.append("Showed solution orientation.")
        else:
            # Only penalize if there was also no contextual negatives (otherwise that system handles it)
            if not found_blunt_negative and not found_contextual_negative:
                score -= 1.0
                areas_for_improvement.append("When delivering difficult news or if a direct solution isn't available, try to frame positively or explicitly offer alternatives (e.g., 'What I can do is...' or 'Let's explore options').")
                feedback_log.append("Could improve positive framing or solution focus.")
        
        # 3. Check for concrete actions or next steps
        action_patterns = [
            r'(I will|I\'ll|We will|We\'ll) (\w+)',  # Promises of specific actions
            r'(Here are|Let me outline) (the next steps|what happens next)',  # Process explanations
            r'(You can expect|You should receive) .+ (by|within|in) .+',  # Timing expectations
            r'(please let me know|reach out|contact us)',  # Invitation for further engagement
            r'(I\'d be happy to|I\'d love to) help',  # Offers of assistance
        ]
        
        action_found = False
        for pattern in action_patterns:
            if re.search(pattern, text_to_analyze, re.IGNORECASE):
                if not action_found:  # Only add the score once
                    score += 1.0
                    action_found = True
                strengths.append("Provided concrete next steps or offers of assistance.")
                feedback_log.append("Included specific action steps or offers of help.")
                break
                
        # 4. Special handling for refund denial - check if policy is explained
        if "refund" in text_to_analyze.lower() and ("denied" in text_to_analyze.lower() or "denial" in text_to_analyze.lower() or "unable to" in text_to_analyze.lower()):
            if "policy" in text_to_analyze.lower() and re.search(r'(our policy|refund policy|guidelines)', text_to_analyze, re.IGNORECASE):
                score += 1.0
                strengths.append("Explained refund policy to support the decision.")
                feedback_log.append("Provided policy explanation for refund denial.")
        
        return round(max(0.0, min(10.0, score)), 1)

    def _evaluate_clarity_and_confidence(self, text_to_analyze: str, strengths: list[str], areas_for_improvement: list[str], feedback_log: list[str]) -> float:
        """Evaluates clarity of language and advisor's confidence."""
        score = 5.0

        # 1. Check for Uncertainty Phrases
        found_uncertainty, uncertain_phrases = self._check_phrases(text_to_analyze, UNCERTAINTY_PHRASES)
        if found_uncertainty:
            # Check if uncertainty is about solution (bad) or just offering flexibility (okay)
            is_flexibility = False
            for phrase in uncertain_phrases:
                context = self._get_phrase_context(text_to_analyze, phrase, window=50)
                if any(flex_term in context.lower() for flex_term in ["options", "alternative", "solution", "help", "assist", "troubleshoot"]):
                    is_flexibility = True
                    break
                    
            if is_flexibility:
                # No penalty for uncertainty when offering flexible solutions
                feedback_log.append("Used uncertainty phrases in context of offering flexibility.")
            else:
                score -= 2.0
                for phrase in uncertain_phrases[:2]:  # Limit to two examples
                    areas_for_improvement.append(f"Language suggested uncertainty (e.g., '{phrase}'). Aim for confident and clear statements. If unsure, state what you will do to find out.")
                feedback_log.append("Expressed uncertainty.")
        else:
            score += 0.5
            feedback_log.append("Avoided strong uncertainty phrases.")
        
        # 1b. Check for flexibility phrases (may not be penalized)
        found_flexibility, flexibility_phrases = self._check_phrases(text_to_analyze, FLEXIBILITY_PHRASES)
        if found_flexibility:
            # Check if flexibility phrases are about solutions or options
            is_solution_context = False
            for phrase in flexibility_phrases:
                context = self._get_phrase_context(text_to_analyze, phrase, window=50)
                if any(solution_term in context.lower() for solution_term in ["options", "alternative", "solution", "help", "suggest", "offer", "available"]):
                    is_solution_context = True
                    break
                    
            if is_solution_context:
                # No penalty - actually a bonus - for flexibility when offering solutions
                score += 0.5
                strengths.append("Used appropriately flexible language when discussing possible solutions.")
                feedback_log.append("Used flexibility phrases appropriately in solution context.")
            else:
                # Mild penalty for flexibility phrases in non-solution contexts
                score -= 0.5
                areas_for_improvement.append(f"Phrase '{flexibility_phrases[0]}' suggests uncertainty. Consider more confident language except when discussing options or alternatives.")
                feedback_log.append("Used flexibility phrases in non-solution context.")
                
        # 1c. Check for appropriate caution phrases (positive)
        found_caution, caution_phrases = self._check_phrases(text_to_analyze, APPROPRIATE_CAUTION_PHRASES)
        if found_caution:
            score += 0.5
            strengths.append(f"Used appropriate qualifiers like '{caution_phrases[0]}' to avoid overgeneralizing.")
            feedback_log.append("Used appropriate caution phrases.")

        # 2. Check for less confident phrases
        found_less_confident, less_conf_phrases = self._check_phrases(text_to_analyze, LESS_CONFIDENT_PHRASES)
        if found_less_confident:
            # Check if less confident phrases are used in customer service context offering help
            is_help_context = False
            for phrase in less_conf_phrases:
                context = self._get_phrase_context(text_to_analyze, phrase, window=50)
                if any(help_term in context.lower() for help_term in ["help you", "assist you", "support", "check for you"]):
                    is_help_context = True
                    break
                    
            if is_help_context:
                # No penalty for less confident phrases in help context
                feedback_log.append("Used less confident phrases appropriately when offering help.")
            else:
                score -= 0.5  # Reduced penalty 
                for phrase in less_conf_phrases[:1]:  # Just one example
                    areas_for_improvement.append(f"Phrasing like '{phrase}' can be improved for more assurance. For example, instead of 'I'll have to check', try 'Let me verify that for you quickly.'.")
                feedback_log.append("Used less confident phrasing.")
        
        # 3. Check for clear structure and organization
        # Look for paragraph breaks, bullet points, numbered lists, etc.
        paragraphs = [p for p in text_to_analyze.split('\n') if p.strip()]
        if len(paragraphs) >= 3:  # More than 2 paragraphs
            score += 1.5  # Increased reward for good structure
            strengths.append("Used clear paragraph structure for organization.")
            feedback_log.append("Good text organization with paragraphs.")
            
            # Additional check for logical flow of paragraphs (intro, body, conclusion)
            has_greeting = re.search(r'^(Hello|Hi|Good (morning|afternoon|evening)|Greetings|Dear)', paragraphs[0], re.IGNORECASE)
            has_closing = re.search(r'(Thank you|Thanks|Best regards|Sincerely|Regards|Yours truly|Appreciate your|Looking forward)', paragraphs[-1], re.IGNORECASE)
            
            if has_greeting and has_closing:
                score += 0.5  # Bonus for complete structure with greeting and closing
                strengths.append("Organized message with clear introduction and conclusion.")
        
        if re.search(r'(\d+\.|\*|\-)\s', text_to_analyze):  # Numbered or bulleted lists
            score += 1.0
            strengths.append("Used lists to organize information clearly.")
            feedback_log.append("Effective use of lists for clarity.")
        
        # 4. Check for jargon or overly complex language
        avg_word_length = sum(len(word) for word in text_to_analyze.split()) / len(text_to_analyze.split()) if text_to_analyze.split() else 0
        if avg_word_length > 8.5:  # Adjusted threshold for potential jargon/complexity
            score -= 1.0
            areas_for_improvement.append("Consider using simpler language. Some sentences contain complex words that might be difficult to understand.")
            feedback_log.append("Used potentially complex language.")
        
        # 5. Check for sentence length (too long = harder to understand)
        sentences = re.split(r'[.!?]+', text_to_analyze)
        long_sentences = [s for s in sentences if len(s.split()) > 30]  # Increased threshold for very long sentences
        if long_sentences:
            score -= 1.0
            areas_for_improvement.append("Some sentences are very long. Consider breaking them into shorter, clearer statements.")
            feedback_log.append("Contains overly long sentences.")
            
        # 6. Check for placeholders that weren't filled in (like [X days])
        if re.search(r'\[\w+\s*\w*\]', text_to_analyze):
            score -= 1.0  # Reduced penalty
            areas_for_improvement.append("Response contains unfilled placeholders (like [X]). Ensure all template fields are completed with specific information.")
            feedback_log.append("Contains unfilled placeholders.")
            
        # 7. Check if text explains policies or technical concepts clearly
        policy_terms = ["policy", "terms", "guidelines", "criteria", "requirements"]
        has_policy_terms = any(term in text_to_analyze.lower() for term in policy_terms)
        
        if has_policy_terms:
            # Check if policy terms are explained with examples or specific details
            has_explanation = re.search(r'(for example|such as|specifically|if .+, then|within \d+|by \d+)', text_to_analyze, re.IGNORECASE)
            if has_explanation:
                score += 1.0
                strengths.append("Clearly explained policies with specific details.")
                feedback_log.append("Provided clear policy explanations.")
            elif "website" in text_to_analyze.lower() or "link" in text_to_analyze.lower() or ".com" in text_to_analyze.lower():
                score += 0.5
                strengths.append("Provided a reference to detailed policy information.")
        
        # 8. Additional bonus for responses with both "what" and "why" explanations
        has_what = re.search(r'(we (can\'t|cannot)|unable to|isn\'t possible)', text_to_analyze, re.IGNORECASE)
        has_why = re.search(r'(because|since|as|due to|reason)', text_to_analyze, re.IGNORECASE)
        
        if has_what and has_why:
            score += 0.5
            strengths.append("Explained both what the situation is and why it exists, enhancing clarity.")
            
        return round(max(0.0, min(10.0, score)), 1)

    def _get_phrase_context(self, text: str, phrase: str, window: int = 50) -> str:
        """Get the surrounding context of a phrase in text."""
        idx = text.lower().find(phrase.lower())
        if idx == -1:
            return ""
            
        start = max(0, idx - window)
        end = min(len(text), idx + len(phrase) + window)
        return text[start:end]

    def _evaluate_tone_and_professionalism(self, text_to_analyze: str, strengths: list[str], areas_for_improvement: list[str], feedback_log: list[str]) -> float:
        """Evaluates tone and professionalism (e.g., courtesy)."""
        score = 5.0

        # 1. Check for Courtesy Phrases
        found_courtesy, courtesy_found = self._check_phrases(text_to_analyze, COURTESY_PHRASES)
        found_polite_interj, interj_found = self._check_phrases(text_to_analyze, POLITE_INTERJECTIONS)

        if found_courtesy or found_polite_interj:
            score += 2.0
            if found_courtesy:
                strengths.append(f"Maintained a courteous tone with phrases like '{courtesy_found[0]}'.")
            if found_polite_interj:
                strengths.append(f"Used polite interjections for flow, like '{interj_found[0]}'.")
            feedback_log.append("Used courteous language.")
        else:
            score -= 1.5
            areas_for_improvement.append("Consider incorporating more courtesy phrases (e.g., 'please', 'thank you') to enhance professional tone.")
            feedback_log.append("Lacked explicit courtesy markers.")

        # 2. Check for overly casual language
        casual_markers = ["yeah", "nope", "cool", "awesome", "kinda", "sorta", "btw", "lol", "haha", "guy", "dude"]
        found_casual, casual_words = self._check_phrases(text_to_analyze, casual_markers)
        
        if found_casual:
            score -= 1.5
            areas_for_improvement.append(f"Avoid overly casual language like '{casual_words[0]}' in professional communication.")
            feedback_log.append("Used casual language inappropriate for professional context.")
            
        # 3. Check for appropriate greeting and closing
        has_greeting = re.search(r'^(Hello|Hi|Good (morning|afternoon|evening)|Greetings|Dear)', text_to_analyze)
        has_closing = re.search(r'(Thank you|Thanks|Best regards|Sincerely|Regards|Yours truly|Appreciate your|Looking forward).{1,30}$', text_to_analyze)
        
        if has_greeting and has_closing:
            score += 1.0
            strengths.append("Used professional greeting and closing.")
            feedback_log.append("Complete professional framing with greeting and closing.")
        elif not has_greeting and not has_closing:
            score -= 1.0
            areas_for_improvement.append("Consider adding both a greeting and closing for a more complete professional communication.")
            feedback_log.append("Missing both greeting and closing.")
        elif not has_greeting:
            # Small penalty for missing just one
            score -= 0.5
            areas_for_improvement.append("Consider starting with a greeting to establish rapport.")
            feedback_log.append("Missing greeting.")
        elif not has_closing:
            score -= 0.5
            areas_for_improvement.append("Consider ending with a professional closing.")
            feedback_log.append("Missing closing.")
        
        return round(max(0.0, min(10.0, score)), 1)

    def evaluate_response(self, response: Response, customer_info: dict = None, query_text: str = None) -> Evaluation:
        """
        Evaluates a single advisor response against predefined communication criteria.
        Enhanced to cover more aspects from communication research.
        
        Args:
            response: The response to evaluate
            customer_info: Optional dictionary of customer information for personalization assessment
            query_text: Optional text of the customer's query for context relevance assessment
        """
        text_to_analyze = response.text # Keep original case for now if needed, but helpers use .lower()
        strengths = []
        areas_for_improvement = []
        feedback_log = ["Starting rule-based evaluation..."] # More detailed internal log

        # Initialize score components
        empathy_score = 5.0
        positioning_score = 5.0
        clarity_confidence_score = 5.0
        tone_professionalism_score = 5.0
        personalization_score = 5.0
        context_relevance_score = 5.0
        coherence_score = 5.0
        persuasion_score = 5.0  # Placeholder, to be developed

        # --- First check if the response meets basic coherence/length requirements ---
        coherence_score, coherence_feedback = self._evaluate_response_coherence(text_to_analyze)
        
        # Record findings from coherence analysis
        strengths.extend([f for f in coherence_feedback if "good" in f.lower() or "well" in f.lower()])
        areas_for_improvement.extend([f for f in coherence_feedback if "good" not in f.lower() and "well" not in f.lower()])
        
        # For extremely short or incoherent responses, significantly penalize all categories
        if coherence_score < 3.0:
            feedback_log.append(f"Response fails basic coherence check (score: {coherence_score}). All category scores will be penalized.")
            # Apply penalties to all dimensions for very poor responses
            penalty_factor = 0.5  # Reduce all scores by half for very poor responses
        else:
            penalty_factor = 1.0  # No penalty for acceptable responses
            
        feedback_log.append(f"Response coherence score: {coherence_score}. Applied penalty factor: {penalty_factor}")

        # --- Evaluate different dimensions --- 
        empathy_score = self._evaluate_empathy_and_rapport(text_to_analyze, strengths, areas_for_improvement, feedback_log) * penalty_factor
        positioning_score = self._evaluate_positioning_and_solutions(text_to_analyze, strengths, areas_for_improvement, feedback_log) * penalty_factor
        clarity_confidence_score = self._evaluate_clarity_and_confidence(text_to_analyze, strengths, areas_for_improvement, feedback_log) * penalty_factor
        tone_professionalism_score = self._evaluate_tone_and_professionalism(text_to_analyze, strengths, areas_for_improvement, feedback_log) * penalty_factor
        
        # Optional assessments based on available data
        if customer_info:
            personalization_score, personalization_feedback = self._evaluate_personalization(text_to_analyze, customer_info)
            personalization_score *= penalty_factor
            strengths.extend([f for f in personalization_feedback if "Consider" not in f])
            areas_for_improvement.extend([f for f in personalization_feedback if "Consider" in f])
            feedback_log.append(f"Personalization assessment completed. Score: {personalization_score}")
        
        if query_text:
            context_relevance_score, context_feedback = self._check_context_relevance(text_to_analyze, query_text)
            context_relevance_score *= penalty_factor
            strengths.extend([f for f in context_feedback if "Low" not in f and "Consider" not in f])
            areas_for_improvement.extend([f for f in context_feedback if "Low" in f or "Consider" in f])
            feedback_log.append(f"Context relevance assessment completed. Score: {context_relevance_score}")

        # --- Calculate Overall Score --- 
        # Weight scores based on importance
        weights = {
            'empathy': 0.20,
            'positioning': 0.20,  # Increased weight for positioning
            'clarity': 0.15,
            'tone': 0.15,
            'coherence': 0.15,
            'personalization': 0.08 if customer_info else 0,
            'context': 0.07 if query_text else 0
        }
        
        # Adjust weights if some dimensions aren't evaluated
        if not customer_info and not query_text:
            weights['empathy'] = 0.25
            weights['positioning'] = 0.25
            weights['clarity'] = 0.20
            weights['tone'] = 0.15
            weights['coherence'] = 0.15
        elif not customer_info:
            weights['empathy'] = 0.22
            weights['positioning'] = 0.22
            weights['clarity'] = 0.18
            weights['tone'] = 0.15
            weights['coherence'] = 0.16
            weights['context'] = 0.07
        elif not query_text:
            weights['empathy'] = 0.22
            weights['positioning'] = 0.22
            weights['clarity'] = 0.15
            weights['tone'] = 0.15
            weights['coherence'] = 0.18
            weights['personalization'] = 0.08
            
        overall_score = (
            weights['empathy'] * empathy_score +
            weights['positioning'] * positioning_score +
            weights['clarity'] * clarity_confidence_score +
            weights['tone'] * tone_professionalism_score +
            weights['coherence'] * coherence_score +
            weights['personalization'] * personalization_score + 
            weights['context'] * context_relevance_score
        )
        
        # Apply a minimal floor for overall score when empathy is very high
        if empathy_score > 8.0 and overall_score < 6.0:
            # Ensure that responses with excellent empathy don't get too low an overall score
            overall_score = max(overall_score, 6.0)
            feedback_log.append(f"Applied minimum overall score floor due to excellent empathy: {overall_score}")
            
        overall_score = round(max(0.0, min(10.0, overall_score)), 1)

        # Consolidate feedback from log for the final Evaluation object
        final_feedback = "Rule-based evaluation highlights: " + " ".join(strengths + areas_for_improvement)
        if not strengths and not areas_for_improvement:
            final_feedback = "Response processed. No specific keyword-based strengths or weaknesses detected by current rules."
        elif not strengths:
            final_feedback = "Response processed. Focus on areas for improvement: " + " ".join(areas_for_improvement)
        elif not areas_for_improvement:
            final_feedback = "Response processed. Good work on: " + " ".join(strengths)

        # Adding narrative feedback summary based on scores
        constructive_summary = []
        if coherence_score < 4.0:
            constructive_summary.append("Response has basic structure/coherence issues. Focus on complete, professional communication.")
        
        if empathy_score < 4.0:
            constructive_summary.append("Empathy could be strengthened.")
        elif empathy_score > 7.0:
            constructive_summary.append("Strong empathetic connection.")
        
        if positioning_score < 4.0:
            constructive_summary.append("Positioning of information needs improvement, especially with negative news.")
        elif positioning_score > 7.0:
            constructive_summary.append("Excellent positioning and solution focus.")

        if clarity_confidence_score < 4.0:
            constructive_summary.append("Clarity and confidence can be enhanced.")
        elif clarity_confidence_score > 7.0:
            constructive_summary.append("Communicated with good clarity and confidence.")

        if tone_professionalism_score < 4.0:
            constructive_summary.append("Professional tone and courtesy could be improved.")
        elif tone_professionalism_score > 7.0:
            constructive_summary.append("Maintained a highly professional and courteous tone.")
            
        if customer_info and personalization_score < 4.0:
            constructive_summary.append("Response could be more personalized to the customer.")
        elif customer_info and personalization_score > 7.0:
            constructive_summary.append("Excellent personalization of the response.")
            
        if query_text and context_relevance_score < 4.0:
            constructive_summary.append("Response could address the query context more directly.")
        elif query_text and context_relevance_score > 7.0:
            constructive_summary.append("Excellent contextual relevance to the query.")

        if constructive_summary:
            final_feedback += " Overall: " + " ".join(constructive_summary)
        else:
            final_feedback += " Overall: Generally balanced communication based on current rules."

        return Evaluation(
            response_id=response.id,
            empathy_score=empathy_score,
            positioning_score=positioning_score,
            persuasion_score=persuasion_score,  # Still a placeholder
            clarity_confidence_score=clarity_confidence_score,
            tone_professionalism_score=tone_professionalism_score,
            personalization_score=personalization_score if customer_info else None,
            context_relevance_score=context_relevance_score if query_text else None,
            coherence_score=coherence_score,
            overall_score=overall_score,
            strengths=list(set(strengths)), # Remove duplicates
            areas_for_improvement=list(set(areas_for_improvement)), # Remove duplicates
            feedback=final_feedback,
            created_at=datetime.now(),
        )

    # We will add more methods here to detect other patterns for:
    # - Persuasion techniques (e.g., yessets)
    # - Identifying summarization of issues
    # - Checking for clarity, tone (simple keyword-based), etc. more deeply
    # - Resilience to spelling errors (e.g. fuzzywuzzy, if allowed)
    # - More nuanced scoring and weighting for overall_score.
    # - Contextual understanding (e.g. was an apology needed and offered?)

    def _update_overall_score(self, score_component: float):
        """Helper to update overall score components."""
        self.total_score_sum += score_component
        self.overall_score_components += 1