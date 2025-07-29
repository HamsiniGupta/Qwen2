import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from collections import Counter
from typing import List, Dict, Optional, Union, Tuple
from rouge_score import rouge_scorer
import warnings
import logging
from pathlib import Path
import os
from datetime import datetime


    # Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class UnifiedMedicalQAEvaluator:
    """
    Unified evaluation framework combining traditional metrics and LLM-based evaluation
    for comprehensive medical QA system assessment.
    """
    
    def __init__(self, 
                 llm_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 enable_llm_eval: bool = True,
                 max_contexts: int = 10):
        """
        Initialize the unified evaluation system.
        
        Args:
            llm_model_name: HuggingFace model for LLM-based evaluation
            embedding_model_name: SentenceTransformer model for semantic similarity
            enable_llm_eval: Whether to enable LLM-based evaluation (requires GPU)
        """
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.enable_llm_eval = enable_llm_eval
        
        # Model instances
        self.llm_tokenizer = None
        self.llm_model = None
        self.embedding_model = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize embedding model (lightweight)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Setup medical normalization patterns
        self._setup_normalization_patterns()

        self.max_contexts = max_contexts
        
        print(f"Unified Medical QA Evaluator initialized")
        print(f"Device: {self.device}")
        print(f"LLM evaluation enabled: {self.enable_llm_eval}")
    
    def _setup_normalization_patterns(self):
        """Setup patterns for medical text normalization."""
        # Age threshold patterns
        self.age_threshold_patterns = [
            (r'(\d+)\s+years?\s+(and|or)\s+older', r'\1 years and older'),
            (r'(?:aged\s+)?[???]\s*(\d+)\s+years?', r'\1 years and older'),
            (r'(?:aged\s+)?(\d+)\+\s+years?', r'\1 years and older'),
            (r'(?:women|female|females|men|male|males)\s+(?:aged\s+)?[???]\s*(\d+)\s+years?', r'aged \1 years and older'),
            (r'(?:women|female|females|men|male|males)\s+(?:aged\s+)?(\d+)\+\s+years?', r'aged \1 years and older'),
            (r'(?:women|female|females|men|male|males)\s+(?:aged\s+)?(\d+)\s+years?\s+(and|or)\s+older', r'aged \1 years and older')
        ]
        
        # Age range patterns
        self.age_range_patterns = [
            (r'(\d+)[-??](\d+)\s+years?\s*(old)?', r'\1 to \2 years'),
            (r'aged\s+(\d+)[-??](\d+)\s+years?\s*(old)?', r'aged \1 to \2 years'),
            (r'(men|male|males|women|female|females)\s+(\d+)[-??](\d+)\s+years?\s*(old)?', r'\1 aged \2 to \3 years'),
            (r'(men|male|males|women|female|females)\s+aged\s+(\d+)[-??](\d+)\s+years?\s*(old)?', r'\1 aged \2 to \3 years')
        ]
        
        # Medical term mappings
        self.medical_mappings = {
            r'\bultrasound\b': 'ultrasonography',
            r'\baaa\b': 'abdominal aortic aneurysm',
            r'\bchd\b': 'coronary heart disease',
            r'\bcvd\b|\bcv\b': 'cardiovascular disease',
            r'\bascvd\b': 'atherosclerotic cardiovascular disease',
            r'\b(high blood pressure|elevated blood pressure)\b': 'hypertension',
            r'\b(high cholesterol|abnormal cholesterol|lipid disorder)\b': 'dyslipidemia',
            r'\b(dm|diabetes mellitus)\b': 'diabetes',
            r'\b(tobacco use|cigarette use)\b': 'smoking',
            r'\bacc\s*[/\\]\s*aha\b|\bacc\s+aha\b|\baccaha\b': 'acc aha',
            r'\bamerican college of cardiology\s*[/\\]\s*american heart association\b': 'acc aha',
            r'\bacc\b': 'american college of cardiology',
            r'\baha\b': 'american heart association',
            r'\bmen\b|\bmale\b|\bmales\b': 'male',
            r'\bwomen\b|\bfemale\b|\bfemales\b': 'female',
            r'\bsiblings?\b': 'sibling',
            r'\boffsprings?\b': 'offspring',
        }

    def load_llm_model(self):
        """Load the LLM model for LLM-based evaluation."""
        if not self.enable_llm_eval:
            print("LLM evaluation disabled, skipping model loading")
            return
            
        try:
            print(f"Loading LLM model: {self.llm_model_name}")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name,
                trust_remote_code=True
            )
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
            print("LLM model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            self.enable_llm_eval = False
            print("Disabling LLM evaluation due to model loading error")

    def normalize(self, text: str) -> str:
        """Comprehensive normalization for medical guidelines text."""
        if text is None or pd.isna(text):
            return ""
        
        original_text = str(text).lower()
        
        # Basic cleaning
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        cleaned_text = ''.join(char for char in original_text if char not in exclude)
        cleaned_text = re.sub(r"\b(a|an|the)\b", " ", cleaned_text)
        cleaned_text = ' '.join(cleaned_text.split())
        
        normalized_text = original_text
        
        # Detect gender
        gender_detected = None
        if re.search(r'\b(?:women|female|females)\b', original_text, re.IGNORECASE):
            gender_detected = 'female'
        elif re.search(r'\b(?:men|male|males)\b', original_text, re.IGNORECASE):
            gender_detected = 'male'
        
        # Process age thresholds
        for pattern, replacement in self.age_threshold_patterns:
            if re.search(pattern, original_text, re.IGNORECASE):
                match = re.search(pattern, original_text, re.IGNORECASE)
                age = match.group(1)
                normalized_text = f"{age} years and older"
                if gender_detected:
                    normalized_text = f"{gender_detected} {normalized_text}"
                break
        
        # Process age ranges if no threshold matched
        if normalized_text == original_text:
            for pattern, replacement in self.age_range_patterns:
                if re.search(pattern, normalized_text, re.IGNORECASE):
                    normalized_text = re.sub(pattern, replacement, normalized_text, flags=re.IGNORECASE)
                    break
        
        # Apply medical term mappings
        for pattern, replacement in self.medical_mappings.items():
            normalized_text = re.sub(pattern, replacement, normalized_text)
        
        # Final cleanup
        normalized_text = re.sub(r'(\d+)\s*[-??]\s*(\d+)\s*years?\s*(old)?', r'\1 to \2 years', normalized_text)
        normalized_text = ' '.join(normalized_text.split())
        
        return normalized_text

    def is_valid_answer(self, answer: str) -> bool:
        # """Check if an answer is valid for evaluation."""
        # if not answer:
        #     return False
        
        # answer_str = str(answer).strip()
        # invalid_values = ['', 'nan', 'null', 'n/a', 'na']
        
        # if answer_str.lower() in invalid_values:
        #     return False
        
        # if len(answer_str.replace('\n', '').replace('\t', '').replace(' ', '')) == 0:
        #     return False
        
        # if len(answer_str) < 3:
        #     return False
        
        return True

    # Traditional Evaluation Methods
    def calculate_rouge_scores(self, answer: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE scores between answer and ground truth."""
        answer_normalized = self.normalize(answer)
        gt_normalized = self.normalize(ground_truth)
        
        if not answer_normalized or not gt_normalized:
            return {f'{metric}_{score_type}': 0.0 
                   for metric in ['rouge1', 'rouge2', 'rougeL'] 
                   for score_type in ['f1', 'precision', 'recall']}
        
        scores = self.rouge_scorer.score(gt_normalized, answer_normalized)
        
        return {
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_f1': scores['rougeL'].fmeasure,
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall
        }

    def fuzzy_match(self, s1: str, s2: str) -> bool:
        """Check if two strings match after normalization."""
        s1_norm = self.normalize(s1)
        s2_norm = self.normalize(s2)
        
        if not s1_norm or not s2_norm:
            return s1_norm == s2_norm
        
        return s1_norm in s2_norm or s2_norm in s1_norm

    def partial_phrase_match(self, s1: str, s2: str, min_common_words: int = 2) -> bool:
        """Check if two strings share significant number of words."""
        s1_norm = self.normalize(s1)
        s2_norm = self.normalize(s2)
        
        if not s1_norm or not s2_norm:
            return False
        
        s1_words = set(s1_norm.split())
        s2_words = set(s2_norm.split())
        common_words = s1_words.intersection(s2_words)
        
        return len(common_words) >= min_common_words

    def list_match_score(self, answer: str, ground_truth: str) -> float:
        """Calculate matching score for lists of items like medical risk factors."""
        answer_norm = self.normalize(answer)
        ground_truth_norm = self.normalize(ground_truth)
        
        if not answer_norm or not ground_truth_norm:
            return 0.0
        
        # Check if this is actually a list
        if ',' not in answer_norm and ',' not in ground_truth_norm and \
           ' and ' not in answer_norm and ' and ' not in ground_truth_norm:
            return 1.0 if self.fuzzy_match(answer, ground_truth) else 0.0
        
        # Split by commas and 'and'
        answer_items = [item.strip() for item in re.split(r',|\sand\s', answer_norm) if item.strip()]
        gt_items = [item.strip() for item in re.split(r',|\sand\s', ground_truth_norm) if item.strip()]
        
        # Match individual items
        matched_items = 0
        for a_item in answer_items:
            for gt_item in gt_items:
                if self.fuzzy_match(a_item, gt_item) or \
                   self.partial_phrase_match(a_item, gt_item, min_common_words=1):
                    matched_items += 1
                    break
        
        # Calculate F1 score
        precision = matched_items / len(answer_items) if answer_items else 0
        recall = matched_items / len(gt_items) if gt_items else 0
        
        if precision + recall == 0:
            return 0
        
        return (2 * precision * recall) / (precision + recall)
        
    def f1_score(self, answer: str, ground_truth: str) -> float:
        """Calculate F1 score based on word overlap."""
        answer_tokens = self.normalize(answer).split()
        ground_truth_tokens = self.normalize(ground_truth).split()
        
        common = Counter(answer_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0
        
        precision = num_same / len(answer_tokens) if answer_tokens else 0
        recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0
        
        if precision + recall == 0:
            return 0
        
        return (2 * precision * recall) / (precision + recall)

    def semantic_similarity(self, answer: str, ground_truth: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        answer_text = self.normalize(answer)
        ground_truth_text = self.normalize(ground_truth)
        
        if not answer_text or not ground_truth_text:
            return 0.0
        
        embeddings = self.embedding_model.encode([answer_text, ground_truth_text])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return similarity
        
    def detect_context_columns(self, df: pd.DataFrame) -> List[str]:
        """Dynamically detect context columns in the dataframe."""
        context_columns = []
        for col in df.columns:
            if col.lower().startswith('context') and col.lower() != 'context':
                context_columns.append(col)
        
        # Sort to ensure consistent ordering (Context1, Context2, Context10, etc.)
        context_columns.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
        
        return context_columns
        
    def check_context_contains_answer(self, context: str, ground_truth: str) -> bool:
        """Check if context contains the ground truth answer."""
        if context is None or ground_truth is None:
            return False
        
        normalized_context = self.normalize(context)
        normalized_gt = self.normalize(ground_truth)
        
        return bool(normalized_gt and normalized_gt in normalized_context)

    # LLM-based Evaluation Methods
    def create_answer_relevancy_prompt(self, question: str, answer: str) -> str:
        """Create prompt for answer relevancy evaluation."""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert evaluator. Your task is to evaluate how well the given answer directly and completely addresses the question.

Provide ONLY a score between 0.0 and 1.0 and brief reasoning in this exact format:
Score: X.X
Reasoning: [Brief explanation]<|eot_id|><|start_header_id|>user<|end_header_id|>

**Question:** {question}

**Answer:** {answer}

**Evaluation Criteria:**
- 1.0: Perfectly relevant, directly and completely addresses the question
- 0.7-0.9: Mostly relevant with minor gaps or slight irrelevant content
- 0.5-0.6: Somewhat relevant but incomplete or contains irrelevant information
- 0.3-0.4: Partially relevant but misses key aspects of the question
- 0.0-0.2: Completely irrelevant or does not address the question at all

Provide your evaluation:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def create_context_adherence_prompt(self, question: str, contexts: List[str], answer: str) -> str:
        """Modified to handle dynamic number of contexts."""
        context_section = ""
        for i, context in enumerate(contexts, 1):
            context_section += f"**Context {i}**: {context}\n\n"
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert evaluator. Evaluate if the given answer is based on and supported by the provided contexts.

Provide ONLY a score between 0.0 and 1.0 and brief reasoning in this exact format:
Score: X.X
Reasoning: [Brief explanation]<|eot_id|><|start_header_id|>user<|end_header_id|>

**Question**: {question}

{context_section}**Generated Answer**: {answer}

**Evaluation Criteria**:
- 1.0: Answer is fully supported by the contexts with no contradictions
- 0.8-0.9: Answer is mostly supported with minor reasonable inferences
- 0.6-0.7: Answer is partially supported but contains some unsupported information
- 0.4-0.5: Answer has limited support and contains significant unsupported information
- 0.2-0.3: Answer is minimally supported and mostly contains external information
- 0.0-0.1: Answer contradicts the contexts or is completely unsupported

Provide your evaluation:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def create_context_relevancy_prompt(self, question: str, contexts: List[str]) -> str:
        """Modified to handle dynamic number of contexts."""
        context_section = ""
        scoring_format = ""
        
        for i, context in enumerate(contexts, 1):
            context_section += f"**Context {i}**: {context}\n\n"
            scoring_format += f"CONTEXT {i}: X.X\n"
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert evaluator. Evaluate how well each context can answer the specific question asked.

Provide ONLY the scores and rationale in this exact format:
{scoring_format}RATIONALE: [Brief explanation]<|eot_id|><|start_header_id|>user<|end_header_id|>

**Question**: {question}

{context_section}**Scoring Guidelines**:
- 1.0: Perfect - Context directly and completely addresses the question
- 0.8-0.9: Very Good - Context addresses most aspects with high relevancy
- 0.6-0.7: Good - Context is relevant but may miss some aspects
- 0.4-0.5: Fair - Context has some relevance but significant gaps
- 0.2-0.3: Poor - Context barely addresses the question
- 0.0-0.1: Completely Irrelevant - Context does not address the question

Provide your evaluation:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def generate_llm_evaluation(self, prompt: str) -> Dict[str, any]:
        """Generate evaluation using the loaded LLM model."""
        if not self.enable_llm_eval or not self.llm_model:
            return {"raw_response": "LLM evaluation disabled"}
            
        try:
            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=3500,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.01,
                    do_sample=False,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = self.llm_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return {"raw_response": response}
            
        except Exception as e:
            logger.error(f"Error generating LLM evaluation: {e}")
            return {"raw_response": f"Error in evaluation: {str(e)}"}

    def parse_llm_response(self, response: str, response_type: str, num_contexts: int = None) -> Dict[str, any]:
        """Parse LLM evaluation responses with robust regex patterns."""
        try:
            if response_type in ['answer_relevancy', 'context_adherence']:
                # Single score extraction
                score_patterns = [
                    r'Score:\s*([0-9]*\.?[0-9]+)',
                    r'score:\s*([0-9]*\.?[0-9]+)',
                    r'Score\s*=\s*([0-9]*\.?[0-9]+)',
                    r'([0-9]*\.?[0-9]+)/1\.0',
                    r'([0-9]*\.?[0-9]+)\s*out\s*of\s*1'
                ]
                
                score = 0.0
                for pattern in score_patterns:
                    score_match = re.search(pattern, response, re.IGNORECASE)
                    if score_match:
                        score = float(score_match.group(1))
                        break
                
                if score == 0.0:
                    first_number = re.search(r'([0-9]*\.?[0-9]+)', response)
                    if first_number:
                        potential_score = float(first_number.group(1))
                        if 0.0 <= potential_score <= 1.0:
                            score = potential_score
                
                score = max(0.0, min(1.0, score))
                
                # Extract reasoning
                reasoning_patterns = [
                    r'Reasoning:\s*(.+)',
                    r'reasoning:\s*(.+)',
                    r'Explanation:\s*(.+)',
                    r'explanation:\s*(.+)'
                ]
                
                reasoning = ""
                for pattern in reasoning_patterns:
                    reasoning_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                        break
                
                if not reasoning:
                    reasoning = response[:200] + "..." if len(response) > 200 else response
                
                return {
                    "score": score,
                    "reasoning": reasoning,
                    "raw_response": response
                }
                
            elif response_type == 'context_relevancy':
                # If num_contexts not provided, try to detect from the response
                if num_contexts is None:
                    # Count how many "CONTEXT X:" patterns appear in the response
                    context_matches = re.findall(r'CONTEXT\s+(\d+):', response, re.IGNORECASE)
                    if context_matches:
                        num_contexts = max(int(match) for match in context_matches)
                    else:
                        num_contexts = 2  # Default fallback
               
                context_scores = {}
            
                # Generate patterns for each context dynamically
                for i in range(1, num_contexts + 1):
                    context_patterns = [
                        f'CONTEXT {i}:\\s*([0-9]*\\.?[0-9]+)',
                        f'Context {i}:\\s*([0-9]*\\.?[0-9]+)',
                        f'context {i}:\\s*([0-9]*\\.?[0-9]+)',
                        f'CONTEXT{i}:\\s*([0-9]*\\.?[0-9]+)'
                    ]
                
                    context_score = 0.0
                    for pattern in context_patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            context_score = float(match.group(1))
                            break
                
                    context_scores[f'context{i}_score'] = max(0.0, min(1.0, context_score))
            
                # If no scores found, try to extract from sequential numbers
                if all(score == 0.0 for score in context_scores.values()):
                    numbers = re.findall(r'([0-9]*\\.?[0-9]+)', response)
                    valid_scores = [float(n) for n in numbers if 0.0 <= float(n) <= 1.0]
                
                    for i, score in enumerate(valid_scores[:num_contexts], 1):
                        context_scores[f'context{i}_score'] = score
            
                # Calculate average score
                scores_list = list(context_scores.values())
                avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
            
                # Extract rationale
                rationale_patterns = [
                    r'RATIONALE:\\s*(.+)',
                    r'Rationale:\\s*(.+)',
                    r'rationale:\\s*(.+)',
                    r'EXPLANATION:\\s*(.+)',
                    r'Explanation:\\s*(.+)'
                ]
            
                rationale = ""
                for pattern in rationale_patterns:
                    rationale_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                    if rationale_match:
                        rationale = rationale_match.group(1).strip()
                        break
            
                if not rationale:
                    rationale = response[:200] + "..." if len(response) > 200 else response
            
                # Build result dictionary with dynamic context scores
                result = {
                    "avg_score": avg_score,
                    "rationale": rationale,
                    "raw_response": response
                }
            
                # Add individual context scores
                result.update(context_scores)
            
                # For backwards compatibility with existing code that expects specific keys
                if num_contexts >= 1:
                    result["context1_score"] = context_scores.get('context1_score', 0.0)
                if num_contexts >= 2:
                    result["context2_score"] = context_scores.get('context2_score', 0.0)
            
                return result

        except Exception as e:
            if response_type == 'context_relevancy':
                # Return default structure for any number of contexts
                result = {
                    "avg_score": 0.0,
                    "rationale": f"Error parsing response: {str(e)}",
                    "raw_response": response
                }
            
                # Add dynamic context scores
                for i in range(1, num_contexts + 1):
                    result[f'context{i}_score'] = 0.0
            
                # For backwards compatibility
                if num_contexts >= 1:
                    result["context1_score"] = 0.0
                if num_contexts >= 2:
                    result["context2_score"] = 0.0
            
                return result
            else:
                return {
                    "score": 0.0,
                    "reasoning": f"Error parsing response: {str(e)}",
                    "raw_response": response
                }

    # Comprehensive Evaluation Methods
    def calculate_retrieval_metrics(self, df: pd.DataFrame, 
                                  context_columns: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Modified to handle dynamic context columns."""
        if context_columns is None:
            context_columns = self.detect_context_columns(df)
        
        result_df = df.copy()
        
        retrieval_recalls = []
        retrieval_precisions = []
        average_precisions = []
        context_hits = []
        
        for _, row in df.iterrows():
            ground_truth = row['Ground_truth']
            
            hits = {}
            valid_contexts = 0
            relevant_positions = []
            
            for i, context_col in enumerate(context_columns):
                if context_col in row and not pd.isna(row[context_col]):
                    context = row[context_col]
                    contains_answer = self.check_context_contains_answer(context, ground_truth)
                    hits[f'{context_col}_contains_answer'] = contains_answer
                    valid_contexts += 1
                    
                    if contains_answer:
                        relevant_positions.append(i + 1)
                else:
                    hits[f'{context_col}_contains_answer'] = False
            
            context_hits.append(hits)
            
            hit_list = [hits[f'{col}_contains_answer'] for col in context_columns 
                       if f'{col}_contains_answer' in hits]
            
            # Recall: 1 if any context contains answer
            recall = 1 if any(hit_list) else 0
            retrieval_recalls.append(recall)
            
            # Precision: fraction of contexts that contain answer
            precision = sum(hit_list) / valid_contexts if valid_contexts > 0 else 0.0
            retrieval_precisions.append(precision)
            
            # Average Precision calculation remains the same
            if not relevant_positions:
                ap_score = 0.0
            else:
                precision_at_relevant = []
                for pos in relevant_positions:
                    relevant_at_or_before = sum(1 for rel_pos in relevant_positions if rel_pos <= pos)
                    precision_at_pos = relevant_at_or_before / pos
                    precision_at_relevant.append(precision_at_pos)
                ap_score = sum(precision_at_relevant) / len(precision_at_relevant)
            
            average_precisions.append(ap_score)
        
        # Add metrics to dataframe
        result_df['retrieval_recall'] = retrieval_recalls
        result_df['retrieval_precision'] = retrieval_precisions
        result_df['average_precision'] = average_precisions
        
        # Add individual context hit information
        for context_col in context_columns:
            col_name = f'{context_col}_contains_answer'
            result_df[col_name] = [hit.get(col_name, False) for hit in context_hits]
        
        # Calculate overall metrics
        overall_metrics = {
            'retrieval_recall': np.mean(retrieval_recalls),
            'retrieval_precision': np.mean(retrieval_precisions),
            'mean_average_precision': np.mean(average_precisions)
        }
        
        return result_df, overall_metrics

    def evaluate_single_row(self, row_data: Dict, evaluation_types: List[str]) -> Dict:
        """Modified to handle dynamic contexts."""
        # Detect context columns dynamically
        context_keys = [k for k in row_data.keys() if k.lower().startswith('context') and k.lower() != 'context']
        context_keys.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
        
        results = {
            'question': row_data.get('question', ''),
            'Answer': row_data.get('Answer', ''),
            'Ground_truth': row_data.get('Ground_truth', '')
        }
        
        # Add all context columns to results
        for context_key in context_keys:
            results[context_key] = row_data.get(context_key, '')
        
        answer_is_valid = self.is_valid_answer(row_data.get('Answer', ''))
        
        # Traditional metrics
        if 'traditional' in evaluation_types:
            if answer_is_valid:
                # ROUGE scores
                rouge_scores = self.calculate_rouge_scores(
                    row_data.get('Answer', ''), 
                    row_data.get('Ground_truth', '')
                )
                results.update(rouge_scores)
                
                # F1 score
                results['f1_score'] = self.f1_score(
                    row_data.get('Answer', ''), 
                    row_data.get('Ground_truth', '')
                )

                # List match score
                results['list_match_score'] = self.list_match_score(
                    row_data.get('Answer', ''), 
                    row_data.get('Ground_truth', '')
                )
                
                # Combined score (best of F1 and list match)
                results['combined_score'] = max(
                    results['f1_score'], 
                    results['list_match_score']
                )
                
                # Semantic similarity
                results['semantic_similarity'] = self.semantic_similarity(
                    row_data.get('Answer', ''), 
                    row_data.get('Ground_truth', '')
                )
                
                # Exact and fuzzy match
                results['exact_match'] = (
                    self.normalize(row_data.get('Answer', '')) == 
                    self.normalize(row_data.get('Ground_truth', ''))
                )
                results['fuzzy_match'] = self.fuzzy_match(
                    row_data.get('Answer', ''), 
                    row_data.get('Ground_truth', '')
                )
            else:
                # Set default values for invalid answers
                rouge_metrics = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1', 
                               'rouge1_precision', 'rouge1_recall',
                               'rouge2_precision', 'rouge2_recall',
                               'rougeL_precision', 'rougeL_recall']
                for metric in rouge_metrics:
                    results[metric] = None
                
                results.update({
                    'f1_score': None,
                    'list_match_score': None,
                    'combined_score': None,
                    'semantic_similarity': None,
                    'exact_match': False,
                    'fuzzy_match': False
                })
        
        # LLM-based evaluation
        if 'llm' in evaluation_types and self.enable_llm_eval:
            if 'answer_relevancy' in evaluation_types or 'llm' in evaluation_types:
                if not answer_is_valid:
                    results.update({
                        'answer_relevancy_score': None,
                        'answer_relevancy_reasoning': "No valid answer provided - evaluation not applicable",
                        'answer_relevancy_raw': "SKIPPED: No valid answer to evaluate"
                    })
                else:
                    try:
                        prompt = self.create_answer_relevancy_prompt(
                            row_data.get('question', ''), 
                            row_data.get('Answer', '')
                        )
                        response = self.generate_llm_evaluation(prompt)
                        parsed = self.parse_llm_response(response['raw_response'], 'answer_relevancy')
                        
                        results.update({
                            'answer_relevancy_score': parsed['score'],
                            'answer_relevancy_reasoning': parsed['reasoning'],
                            'answer_relevancy_raw': parsed['raw_response']
                        })
                    except Exception as e:
                        results.update({
                            'answer_relevancy_score': None,
                            'answer_relevancy_reasoning': f"Error: {str(e)}",
                            'answer_relevancy_raw': ""
                        })
            
            if 'context_adherence' in evaluation_types or 'llm' in evaluation_types:
                if not answer_is_valid:
                    results.update({
                        'context_adherence_score': None,
                        'context_adherence_reasoning': "No valid answer provided - evaluation not applicable",
                        'context_adherence_raw': "SKIPPED: No valid answer to evaluate"
                    })
                else:
                    try:
                        contexts = [row_data.get(key, '') for key in context_keys]
                        prompt = self.create_context_adherence_prompt(
                            row_data.get('question', ''),
                            contexts,
                            row_data.get('Answer', '')
                        )
                        response = self.generate_llm_evaluation(prompt)
                        parsed = self.parse_llm_response(response['raw_response'], 'context_adherence')
                        
                        results.update({
                            'context_adherence_score': parsed['score'],
                            'context_adherence_reasoning': parsed['reasoning'],
                            'context_adherence_raw': parsed['raw_response']
                        })
                    except Exception as e:
                        results.update({
                            'context_adherence_score': None,
                            'context_adherence_reasoning': f"Error: {str(e)}",
                            'context_adherence_raw': ""
                        })
            
            if 'context_relevancy' in evaluation_types or 'llm' in evaluation_types:
                try:
                    contexts = [row_data.get(key, '') for key in context_keys]
                    prompt = self.create_context_relevancy_prompt(
                        row_data.get('question', ''),
                        contexts
                    )
                    response = self.generate_llm_evaluation(prompt)
                    parsed = self.parse_llm_response(response['raw_response'], 'context_relevancy' , num_contexts=len(contexts))
                    
                    # Add individual context scores
                    for i in range(1, len(contexts) + 1):
                        score_key = f'context{i}_score'
                        if score_key in parsed:
                            results[f'context_relevancy_context{i}_score'] = parsed[score_key]
                    
                    results.update({
                        'context_relevancy_avg_score': parsed['avg_score'],
                        'context_relevancy_rationale': parsed['rationale'],
                        'context_relevancy_raw': parsed['raw_response']
                    })
                except Exception as e:
                    # Handle errors for dynamic number of contexts
                    for i in range(1, len(context_keys) + 1):
                        results[f'context_relevancy_context{i}_score'] = 0.0
                    
                    results.update({
                        'context_relevancy_avg_score': 0.0,
                        'context_relevancy_rationale': f"Error: {str(e)}",
                        'context_relevancy_raw': ""
                    })
        
        return results

    def categorize_response(self, row: pd.Series) -> str:
        """Categorize response quality based on various metrics."""
        if row.get('exact_match', False):
            return 'Exact Match'
        elif row.get('fuzzy_match', False):
            return 'Fuzzy Match'
        elif row.get('list_match_score', 0) >= 0.7:
            return 'List Component Match'
        elif row.get('semantic_similarity', 0) >= 0.7:
            return 'High Semantic Similarity'
        elif row.get('semantic_similarity', 0) >= 0.4:
            return 'Moderate Semantic Similarity'
        else:
            return 'Low Similarity'

    def calculate_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate accuracy based on successful match categories."""
        successful_categories = ['Exact Match', 'Fuzzy Match', 'High Semantic Similarity', 'List Component Match']
        if 'match_category' in df.columns:
            return df['match_category'].isin(successful_categories).mean()
        return 0.0

    def analyze_results(self, df: pd.DataFrame) -> Dict:
        """Analyze evaluation results and generate comprehensive metrics."""
        metrics = {}
        
        # Traditional QA metrics
        traditional_metrics = ['f1_score', 'list_match_score', 'combined_score', 'semantic_similarity', 'exact_match', 'fuzzy_match']
        for metric in traditional_metrics:
            if metric in df.columns:
                valid_scores = df[metric].dropna() if metric in ['f1_score', 'semantic_similarity'] else df[metric]
                if len(valid_scores) > 0:
                    metrics[f'avg_{metric}'] = valid_scores.mean()
        
        # ROUGE metrics
        rouge_metrics = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1', 
                        'rouge1_precision', 'rouge1_recall',
                        'rouge2_precision', 'rouge2_recall',
                        'rougeL_precision', 'rougeL_recall']
        for metric in rouge_metrics:
            if metric in df.columns:
                valid_scores = df[metric].dropna()
                if len(valid_scores) > 0:
                    metrics[f'avg_{metric}'] = valid_scores.mean()
        
        # Success rates
        if 'semantic_similarity' in df.columns:
            valid_sim = df['semantic_similarity'].dropna()
            if len(valid_sim) > 0:
                metrics['success_rate_threshold_0.7'] = (valid_sim >= 0.7).mean()
                metrics['success_rate_threshold_0.5'] = (valid_sim >= 0.5).mean()
        
        # LLM-based metrics
        llm_metrics = ['answer_relevancy_score', 'context_adherence_score', 'context_relevancy_avg_score']
        for metric in llm_metrics:
            if metric in df.columns:
                valid_scores = df[metric].dropna()
                if len(valid_scores) > 0:
                    metrics[f'avg_{metric}'] = valid_scores.mean()
                    metrics[f'{metric}_skipped'] = (df[metric].isna()).sum()
        
        # Retrieval metrics
        retrieval_metrics = ['retrieval_recall', 'retrieval_precision', 'average_precision']
        for metric in retrieval_metrics:
            if metric in df.columns:
                metrics[metric] = df[metric].mean()
        
        # MAP performance breakdown
        if 'average_precision' in df.columns:
            metrics['map_perfect_scores'] = (df['average_precision'] == 1.0).mean()
            metrics['map_above_0.8'] = (df['average_precision'] >= 0.8).mean()
            metrics['map_above_0.5'] = (df['average_precision'] >= 0.5).mean()
            metrics['map_zero_scores'] = (df['average_precision'] == 0.0).mean()
        
        # Context-specific metrics
        context_cols = [col for col in df.columns if col.endswith('_contains_answer')]
        for col in context_cols:
            context_name = col.replace('_contains_answer', '')
            metrics[f'{context_name}_hit_rate'] = df[col].mean()
        
        # Combined context analysis

        if len(context_cols) > 0:
            # Any context hit rate (at least one context contains the answer)
            context_hit_series = df[context_cols].any(axis=1)
            metrics['any_context_hit_rate'] = context_hit_series.mean()
    
            # All contexts hit rate (all contexts contain the answer)
            if len(context_cols) > 1:
                all_contexts_hit_series = df[context_cols].all(axis=1)
                metrics['all_contexts_hit_rate'] = all_contexts_hit_series.mean()
        
                # No context hit rate (none of the contexts contain the answer)
                no_context_hit_series = ~df[context_cols].any(axis=1)
                metrics['no_context_hit_rate'] = no_context_hit_series.mean()
        
                # Calculate individual context exclusive hit rates (only this context hits)
                for i, col in enumerate(context_cols):
                    # Create a mask for "only this context hits"
                    other_contexts = [c for j, c in enumerate(context_cols) if j != i]
                    if other_contexts:
                        only_this_context = df[col] & ~df[other_contexts].any(axis=1)
                        context_name = col.replace('_contains_answer', '')
                        metrics[f'only_{context_name}_hit_rate'] = only_this_context.mean()
        
                # Calculate hit distribution (how many contexts hit simultaneously)
                context_hit_counts = df[context_cols].sum(axis=1)
                for i in range(len(context_cols) + 1):
                    count_name = f'{i}_contexts_hit_rate'
                    metrics[count_name] = (context_hit_counts == i).mean()
    
        # Context redundancy analysis (if multiple contexts available)
        if len(context_cols) >= 2:
            # Calculate overlap between all pairs of contexts
            context_names = [col.replace('_contains_answer', '') for col in context_cols]
        
            for i in range(len(context_cols)):
                for j in range(i + 1, len(context_cols)):
                    col1, col2 = context_cols[i], context_cols[j]
                    name1, name2 = context_names[i], context_names[j]
                
                    # Both contexts hit
                    both_hit = (df[col1] & df[col2]).mean()
                    metrics[f'{name1}_{name2}_both_hit_rate'] = both_hit
                
                    # Only first context hits
                    only_first = (df[col1] & ~df[col2]).mean()
                    metrics[f'{name1}_only_hit_rate'] = only_first
                
                    # Only second context hits  
                    only_second = (~df[col1] & df[col2]).mean()
                    metrics[f'{name2}_only_hit_rate'] = only_second
                
        if 'Context1_contains_answer' in df.columns and 'Context2_contains_answer' in df.columns:
            metrics['both_contexts_hit_rate'] = (df['Context1_contains_answer'] & df['Context2_contains_answer']).mean()
            metrics['any_context_hit_rate'] = (df['Context1_contains_answer'] | df['Context2_contains_answer']).mean()
            metrics['only_context1_hit_rate'] = (df['Context1_contains_answer'] & ~df['Context2_contains_answer']).mean()
            metrics['only_context2_hit_rate'] = (~df['Context1_contains_answer'] & df['Context2_contains_answer']).mean()
        
        # Match category breakdown
        if 'match_category' in df.columns:
            category_counts = df['match_category'].value_counts().to_dict()
            for category, count in category_counts.items():
                metrics[f'category_{category.replace(" ", "_").lower()}'] = count / len(df)
        
        # Overall accuracy
        metrics['accuracy'] = self.calculate_accuracy(df)
        
        return metrics

    def complete_evaluation_pipeline(self, 
                               input_file: str, 
                               evaluation_types: List[str] = ['traditional', 'llm'],
                               output_file: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete evaluation pipeline with all metrics.
    
        Args:
        input_file: Path to CSV file
        evaluation_types: List of evaluation types ['traditional', 'llm', 'answer_relevancy', 'context_adherence', 'context_relevancy']
        output_file: Optional output file path
    
        Returns:
            Tuple of (results_dataframe, metrics_dict)
        """
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} questions")

        # Detect context columns dynamically
        context_columns = self.detect_context_columns(df)
        print(f"Detected {len(context_columns)} context columns: {context_columns}")
    
        # Validate that we have the expected context columns
        #if not context_columns:
         #   raise ValueError("No context columns found in the dataset")
        
        # Load LLM model if needed
        if any(eval_type in ['llm', 'answer_relevancy', 'context_adherence', 'context_relevancy'] 
               for eval_type in evaluation_types):
            if self.enable_llm_eval and not self.llm_model:
                self.load_llm_model()
    
        # Check for empty answers
        df['Answer'] = df['Answer'].fillna('')
        empty_answer_count = sum(1 for _, row in df.iterrows() 
                           if not self.is_valid_answer(row.get('Answer', '')))
        print(f"Rows with empty/invalid answers: {empty_answer_count}")
    
        # FIXED: Validate required columns dynamically
        required_columns = ['question']
        if any(eval_type in ['traditional', 'llm', 'answer_relevancy', 'context_adherence'] 
               for eval_type in evaluation_types):
            if 'Answer' not in df.columns:
                raise ValueError("Answer column required for answer-dependent evaluation")
    
        # FIXED: Check for context columns dynamically instead of hardcoding Context1, Context2
        #if any(eval_type in ['llm', 'context_adherence', 'context_relevancy'] 
               #for eval_type in evaluation_types):
            # Instead of hardcoding Context1, Context2, check if we have any context columns
            #if not context_columns:
                #raise ValueError("Context columns required for context-dependent evaluation")
        # Optionally, you can require a minimum number of contexts
        # if len(context_columns) < 2:
        #     raise ValueError("At least 2 context columns required for context evaluation")
    
        #missing_columns = [col for col in required_columns if col not in df.columns]
        #if missing_columns:
         #   raise ValueError(f"Missing required columns: {missing_columns}")
    
        # Process each row
        print("Evaluating questions...")
        results = []
    
        for idx, row in df.iterrows():
            if (idx + 1) % 10 == 0:
                print(f"Processing row {idx + 1}/{len(df)}")
        
            # FIXED: Create row_data dynamically instead of hardcoding Context1, Context2
            row_data = {
                'question': str(row.get('question', '')),
                'Answer': str(row.get('Answer', '')),
                'Ground_truth': str(row.get('Ground_truth', ''))
            }
        
            # Dynamically add all detected context columns
            for context_col in context_columns:
                row_data[context_col] = str(row.get(context_col, ''))
        
            result = self.evaluate_single_row(row_data, evaluation_types)
            results.append(result)
    
        # Create results DataFrame
        results_df = pd.DataFrame(results)
    
        # Add response categorization for traditional metrics
        if 'traditional' in evaluation_types:
            results_df['match_category'] = results_df.apply(self.categorize_response, axis=1)
    
        # Calculate retrieval metrics
        if 'traditional' in evaluation_types:
            print("Calculating retrieval metrics...")
            results_df, retrieval_metrics = self.calculate_retrieval_metrics(results_df, context_columns)
        else:
            retrieval_metrics = {}
    
        # Analyze all results
        print("Analyzing results...")
        all_metrics = self.analyze_results(results_df)
        all_metrics.update(retrieval_metrics)
    
        # Print comprehensive summary
        self.print_comprehensive_summary(results_df, all_metrics, input_file, evaluation_types)
    
        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = input_file.replace('.csv', f'_unified_evaluation_{timestamp}.csv')
    
        results_df.to_csv(output_file, index=False)
        print(f"\nComplete evaluation results saved to: {output_file}")
    
        return results_df, all_metrics

    def print_comprehensive_summary(self, df: pd.DataFrame, all_metrics: Dict, 
                                  input_file: str, evaluation_types: List[str]):
        """Print a comprehensive evaluation summary."""
        print("\n" + "="*80)
        print(f"UNIFIED QA EVALUATION SUMMARY - {input_file}")
        print("="*80)
        print(f"Total Questions: {len(df)}")
        print(f"Evaluation Types: {', '.join(evaluation_types)}")
        print()
        
        # Traditional QA Performance
        if 'traditional' in evaluation_types:
            print("TRADITIONAL QA PERFORMANCE:")
            print("-" * 40)
            
            traditional_display = [
                ('Average F1 Score', 'avg_f1_score'),
                ('Average Semantic Similarity', 'avg_semantic_similarity'),
                ('Exact Match Rate', 'avg_exact_match'),
                ('Fuzzy Match Rate', 'avg_fuzzy_match'),
                ('Success Rate (Similarity ? 0.7)', 'success_rate_threshold_0.7'),
                ('Success Rate (Similarity ? 0.5)', 'success_rate_threshold_0.5')
            ]
            
            for label, key in traditional_display:
                if key in all_metrics:
                    print(f"{label:35s} {all_metrics[key]:.4f}")
            print()
            
            # ROUGE Performance
            if 'avg_rouge1_f1' in all_metrics:
                print("ROUGE PERFORMANCE:")
                print("-" * 40)
                rouge_display = [
                    ('Average ROUGE-1 F1', 'avg_rouge1_f1'),
                    ('Average ROUGE-2 F1', 'avg_rouge2_f1'),
                    ('Average ROUGE-L F1', 'avg_rougeL_f1'),
                    ('ROUGE-1 Precision', 'avg_rouge1_precision'),
                    ('ROUGE-1 Recall', 'avg_rouge1_recall')
                ]
                
                for label, key in rouge_display:
                    if key in all_metrics:
                        print(f"{label:35s} {all_metrics[key]:.4f}")
                print()
            
            # Retrieval Performance
            print("RETRIEVAL PERFORMANCE:")
            print("-" * 40)
            retrieval_display = [
                ('Retrieval Recall', 'retrieval_recall'),
                ('Retrieval Precision', 'retrieval_precision'),
                ('Mean Average Precision (MAP)', 'mean_average_precision')
            ]
            
            for label, key in retrieval_display:
                if key in all_metrics:
                    print(f"{label:35s} {all_metrics[key]:.4f}")
            print()
        
        # LLM-based Performance
        if any(eval_type in ['llm', 'answer_relevancy', 'context_adherence', 'context_relevancy'] 
               for eval_type in evaluation_types):
            print("LLM-BASED EVALUATION PERFORMANCE:")
            print("-" * 40)
            
            llm_display = [
                ('Answer Relevancy Score', 'avg_answer_relevancy_score'),
                ('Context Adherence Score', 'avg_context_adherence_score'),
                ('Context Relevancy Score', 'avg_context_relevancy_avg_score')
            ]
            
            for label, key in llm_display:
                if key in all_metrics:
                    skipped_key = key.replace('avg_', '') + '_skipped'
                    skipped_count = all_metrics.get(skipped_key, 0)
                    print(f"{label:35s} {all_metrics[key]:.4f} ({skipped_count} skipped)")
            print()
        
        # Combined Performance Summary
        print("OVERALL PERFORMANCE SUMMARY:")
        print("-" * 40)
        if 'accuracy' in all_metrics:
            print(f"Overall Accuracy:                   {all_metrics['accuracy']:.4f}")
        
        # Show best performing metric
        performance_metrics = []
        if 'avg_semantic_similarity' in all_metrics:
            performance_metrics.append(('Semantic Similarity', all_metrics['avg_semantic_similarity']))
        if 'avg_answer_relevancy_score' in all_metrics:
            performance_metrics.append(('Answer Relevancy', all_metrics['avg_answer_relevancy_score']))
        if 'avg_context_adherence_score' in all_metrics:
            performance_metrics.append(('Context Adherence', all_metrics['avg_context_adherence_score']))
        
        if performance_metrics:
            best_metric = max(performance_metrics, key=lambda x: x[1])
            print(f"Best Performing Metric:             {best_metric[0]} ({best_metric[1]:.4f})")

    def analyze_error_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze common error patterns in evaluation results."""
        error_analysis = {}
        
        # Low similarity cases
        if 'semantic_similarity' in df.columns:
            valid_sim = df['semantic_similarity'].dropna()
            if len(valid_sim) > 0:
                low_sim_mask = valid_sim < 0.4
                error_analysis['low_similarity_count'] = low_sim_mask.sum()
                error_analysis['low_similarity_rate'] = low_sim_mask.mean()
        
        # Cases where retrieval failed but QA succeeded
        if 'retrieval_recall' in df.columns and 'semantic_similarity' in df.columns:
            valid_sim = df['semantic_similarity'].dropna()
            retrieval_fail_qa_success = (df['retrieval_recall'] == 0) & (valid_sim >= 0.7)
            error_analysis['retrieval_fail_qa_success_count'] = retrieval_fail_qa_success.sum()
        
        # Cases where retrieval succeeded but QA failed
        if 'retrieval_recall' in df.columns and 'semantic_similarity' in df.columns:
            valid_sim = df['semantic_similarity'].dropna()
            retrieval_success_qa_fail = (df['retrieval_recall'] == 1) & (valid_sim < 0.4)
            error_analysis['retrieval_success_qa_fail_count'] = retrieval_success_qa_fail.sum()
        
        # LLM vs Traditional metric disagreement
        if 'answer_relevancy_score' in df.columns and 'semantic_similarity' in df.columns:
            llm_scores = df['answer_relevancy_score'].dropna()
            trad_scores = df['semantic_similarity'].dropna()
            if len(llm_scores) > 0 and len(trad_scores) > 0:
                # Find cases where LLM and traditional metrics disagree significantly
                common_indices = llm_scores.index.intersection(trad_scores.index)
                if len(common_indices) > 0:
                    score_diff = abs(llm_scores[common_indices] - trad_scores[common_indices])
                    high_disagreement = score_diff > 0.3
                    error_analysis['llm_traditional_disagreement_count'] = high_disagreement.sum()
                    error_analysis['llm_traditional_disagreement_rate'] = high_disagreement.mean()
        
        return error_analysis

    def create_evaluation_report(self, results_df: pd.DataFrame, metrics: Dict, 
                               output_path: str = None) -> str:
        """Create a detailed evaluation report."""
        if output_path is None:
            output_path = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_path, 'w') as f:
            f.write("UNIFIED MEDICAL QA EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Questions Evaluated: {len(results_df)}\n\n")
            
            # Traditional Metrics Section
            if any(col.startswith('rouge') for col in results_df.columns):
                f.write("TRADITIONAL METRICS:\n")
                f.write("-" * 20 + "\n")
                for key, value in metrics.items():
                    if key.startswith('avg_') and 'rouge' in key:
                        f.write(f"{key}: {value:.4f}\n")
                f.write("\n")
            
            # LLM Metrics Section
            if any(col.endswith('_score') for col in results_df.columns):
                f.write("LLM-BASED METRICS:\n")
                f.write("-" * 20 + "\n")
                for key, value in metrics.items():
                    if 'relevancy' in key or 'adherence' in key:
                        f.write(f"{key}: {value:.4f}\n")
                f.write("\n")
            
            # Error Analysis
            error_patterns = self.analyze_error_patterns(results_df)
            if error_patterns:
                f.write("ERROR ANALYSIS:\n")
                f.write("-" * 20 + "\n")
                for key, value in error_patterns.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            
            if metrics.get('avg_semantic_similarity', 0) < 0.5:
                f.write("- Low semantic similarity suggests need for better answer generation\n")
            
            if metrics.get('retrieval_recall', 0) < 0.5:
                f.write("- Low retrieval recall suggests need for better context retrieval\n")
            
            if metrics.get('avg_answer_relevancy_score', 0) < 0.7:
                f.write("- Low answer relevancy suggests answers may be off-topic\n")
            
            if metrics.get('avg_context_adherence_score', 0) < 0.7:
                f.write("- Low context adherence suggests answers may contain unsupported information\n")
        
        print(f"Detailed evaluation report saved to: {output_path}")
        return output_path

# Example Usage and Utility Functions
def run_unified_evaluation(input_file: str, 
                         evaluation_types: List[str] = ['traditional', 'llm'],
                         enable_llm: bool = True,
                         output_file: str = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to run unified evaluation.
    
    Args:
        input_file: Path to CSV file
        evaluation_types: Types of evaluation to run
        enable_llm: Whether to enable LLM-based evaluation
        output_file: Optional output file path
    
    Returns:
        Tuple of (results_dataframe, metrics_dict)
    """
    evaluator = UnifiedMedicalQAEvaluator(enable_llm_eval=enable_llm)
    return evaluator.complete_evaluation_pipeline(input_file, evaluation_types, output_file)

def create_combined_comparison_report(comparison_df: pd.DataFrame, 
                                    individual_reports: List[str] = None,
                                    output_path: str = None) -> str:
    """Create a combined comparison report for multiple models."""
    if output_path is None:
        output_path = f"combined_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(output_path, 'w') as f:
        f.write("UNIFIED MEDICAL QA - MODEL COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models Compared: {len(comparison_df)}\n\n")
        
        # Overall comparison table
        f.write("OVERALL COMPARISON:\n")
        f.write("-" * 30 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Best performers by metric
        f.write("BEST PERFORMERS BY METRIC:\n")
        f.write("-" * 30 + "\n")
        
        metrics_to_check = [
            'avg_f1_score', 'avg_semantic_similarity', 'avg_rouge1_f1',
            'avg_answer_relevancy_score', 'avg_context_adherence_score',
            'retrieval_recall', 'mean_average_precision', 'accuracy'
        ]
        
        for metric in metrics_to_check:
            if metric in comparison_df.columns and comparison_df[metric].notna().any():
                best_idx = comparison_df[metric].idxmax()
                best_model = comparison_df.loc[best_idx, 'Model']
                best_score = comparison_df.loc[best_idx, metric]
                f.write(f"{metric:30s}: {best_model} ({best_score:.4f})\n")
        
        f.write("\n")
        
        # Performance analysis
        f.write("PERFORMANCE ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        
        # Calculate average performance across all metrics
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            comparison_df['avg_performance'] = comparison_df[numeric_cols].mean(axis=1, skipna=True)
            best_overall_idx = comparison_df['avg_performance'].idxmax()
            best_overall_model = comparison_df.loc[best_overall_idx, 'Model']
            best_overall_score = comparison_df.loc[best_overall_idx, 'avg_performance']
            
            f.write(f"Best Overall Model: {best_overall_model} (avg: {best_overall_score:.4f})\n")
            
            # Performance ranking
            f.write("\nModel Ranking (by average performance):\n")
            ranked_models = comparison_df.nlargest(len(comparison_df), 'avg_performance')
            for i, (_, row) in enumerate(ranked_models.iterrows(), 1):
                f.write(f"{i}. {row['Model']} (avg: {row['avg_performance']:.4f})\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 30 + "\n")
        
        if 'avg_semantic_similarity' in comparison_df.columns:
            best_sim_model = comparison_df.loc[comparison_df['avg_semantic_similarity'].idxmax(), 'Model']
            f.write(f"- For semantic accuracy: Use {best_sim_model}\n")
        
        if 'retrieval_recall' in comparison_df.columns:
            best_ret_model = comparison_df.loc[comparison_df['retrieval_recall'].idxmax(), 'Model']
            f.write(f"- For retrieval performance: Use {best_ret_model}\n")
        
        if 'avg_answer_relevancy_score' in comparison_df.columns:
            best_rel_model = comparison_df.loc[comparison_df['avg_answer_relevancy_score'].idxmax(), 'Model']
            f.write(f"- For answer relevancy: Use {best_rel_model}\n")
        
        f.write("\n")
        
        # Reference to individual reports
        if individual_reports:
            f.write("INDIVIDUAL DETAILED REPORTS:\n")
            f.write("-" * 30 + "\n")
            for report in individual_reports:
                f.write(f"- {report}\n")
    
    print(f"Combined comparison report saved to: {output_path}")
    return output_path


        # Main execution example
if __name__ == "__main__":
    print("Unified Medical QA Evaluation Framework")
    print("=" * 50)
    
    # Compare multiple models
    file_paths = [
        # "guidelines_QA_results_BioMistral_MedCPTCE1024_BM25_rel.csv",
        # "guidelines_QA_results_BioMistral_MedCPTCE1024_25_rel.csv",
        # "guidelines_QA_results_BioMistral_MedCPTCE1024_50_rel.csv",
        # "guidelines_QA_results_BioMistral_MedCPTCE1024_75_rel.csv",
        # "guidelines_QA_results_BioMistral_MedCPTCE1024_85_rel.csv"
        #"guidelines_QA_results_BioMistral_SimCSE1024_BM25_rel.csv",
        #"guidelines_QA_results_BioMistral_SimCSE1024_25_rel.csv",
        #"guidelines_QA_results_BioMistral_SimCSE1024_50_rel.csv",
        #"guidelines_QA_results_BioMistral_SimCSE1024_75_rel.csv",
        #"guidelines_QA_results_BioMistral_SimCSE1024_85_rel.csv"
        #"guidelines_QA_results_BioMistral_Qwen1024_BM25_rel.csv",
        #"guidelines_QA_results_BioMistral_Qwen1024_25_rel.csv",
        #"guidelines_QA_results_BioMistral_Qwen1024_50_rel.csv",
        #"guidelines_QA_results_BioMistral_Qwen1024_75_rel.csv",
        #"guidelines_QA_results_BioMistral_Qwen1024_85_rel.csv"
        #"guidelines_QA_results_Openbio_MedCPT1024_BM25_rel.csv",
        #"guidelines_QA_results_Openbio_MedCPT1024_25_rel.csv",
        #"guidelines_QA_results_Openbio_MedCPT1024_50_rel.csv",
        #"guidelines_QA_results_Openbio_MedCPT1024_75_rel.csv",
        #"guidelines_QA_results_Openbio_MedCPT1024_85_rel.csv"
        #"guidelines_QA_results_Openbio_SimCSE1024_BM25_rel.csv",
        #"guidelines_QA_results_Openbio_SimCSE1024_25_rel.csv",
        #"guidelines_QA_results_Openbio_SimCSE1024_50_rel.csv",
        #"guidelines_QA_results_Openbio_SimCSE1024_75_rel.csv",
        #"guidelines_QA_results_Openbio_SimCSE1024_85_rel.csv"
        #"guidelines_QA_results_Openbio_Qwen1024_BM25_rel.csv",
        #"guidelines_QA_results_Openbio_Qwen1024_25_rel.csv",
        #"guidelines_QA_results_Openbio_Qwen1024_50_rel.csv",
        #"guidelines_QA_results_Openbio_Qwen1024_75_rel.csv",
        #"guidelines_QA_results_Openbio_Qwen1024_85_rel.csv"
        #"guidelines_QA_results_MedCPTCE_1024_BM25_rel.csv",
        #"guidelines_QA_results_MedCPTCE_1024_25_rel.csv",
        #"guidelines_QA_results_MedCPTCE_1024_50_rel.csv",
        #"guidelines_QA_results_MedCPTCE_1024_75_rel.csv",
        #"guidelines_QA_results_MedCPTCE_1024_85_rel.csv"
        #"medquad_qwen_1024_50.csv"
        #"guidelines_QA_results_SimCSE1024_BM25_top3_rel.csv",
        #"guidelines_QA_results_SimCSE1024_25_top3_rel.csv",
        #"guidelines_QA_results_SimCSE1024_50_top3_rel.csv",
        #"guidelines_QA_results_SimCSE1024_75_top3_rel.csv",
        #"guidelines_QA_results_SimCSE1024_85_top3_rel.csv"
        # "2000/guidelines_QA_results_Qwen2k_BM25_top3_rel.csv",
        # "2000/guidelines_QA_results_Qwen2k_25_top3_rel.csv",
        # "2000/guidelines_QA_results_Qwen2k_50_top3_rel.csv",
        # "2000/guidelines_QA_results_Qwen2k_75_top3_rel.csv",
        # "2000/guidelines_QA_results_Qwen2k_85_top3_rel.csv",
        # "512/guidelines_QA_results_Qwen512_BM25_top3_rel.csv",
        # "512/guidelines_QA_results_Qwen512_25_top3_rel.csv",
        # "512/guidelines_QA_results_Qwen512_50_top3_rel.csv",
        # "512/guidelines_QA_results_Qwen512_75_top3_rel.csv",
        "512/guidelines_QA_results_Qwen512_85_top3_rel.csv",
        "1024/guidelines_QA_results_Qwen1024_BM25_top3_rel.csv",
        "1024/guidelines_QA_results_Qwen1024_25_top3_rel.csv",
        "1024/guidelines_QA_results_Qwen1024_50_top3_rel.csv",
        "1024/guidelines_QA_results_Qwen1024_75_top3_rel.csv",
        "1024/guidelines_QA_results_Qwen1024_85_top3_rel.csv"
        #"guidelines_direct_llm.csv",
        #"zs_LLama-8b-combined_guidelines.csv",
        #"guidelines_QA_results_Qwen1024_75_top5_rel.csv"
        #"guidelines_QA_results_Qwen1024_25.csv" ,
        #"guidelines_QA_results_Qwen512_25.csv",
        #"guidelines_QA_results_Qwen2k_25.csv",
        #"guidelines_QA_results_Qwen2k_25_rel.csv",
        #"guidelines_QA_results_Qwen512_25_rel.csv",
        #"guidelines_QA_results_Qwen1024_25_rel.csv",
        ##"guidelines_QA_results_Qwen1024_50_rel.csv",
        ##"guidelines_QA_results_Qwen512_50_rel.csv",
        ##"guidelines_QA_results_Qwen2k_50_rel.csv",
        ###"guidelines_QA_results_BioMistral_Qwen1024_25_rel.csv",
        #"guidelines_QA_results_BioMistral_Qwen1024_50_rel.csv",
        #"guidelines_QA_results_MedCPTCE_1024_25_rel.csv",
        #"guidelines_QA_results_MedCPTCE_1024_50_rel.csv",
        #"guidelines_QA_results_Openbio_Qwen1024_25_rel.csv",
        #"guidelines_QA_results_Openbio_Qwen1024_50_rel.csv",
        #"guidelines_QA_results_SimCSE1024_25_rel.csv",
        #"guidelines_QA_results_SimCSE1024_50_rel.csv"
    ]
    
    #model_names = ["BioMistral_MedCPTCE1024_BM25","BioMistral_MedCPTCE1024_25","BioMistral_MedCPTCE1024_50","BioMistral_MedCPTCE1024_75","BioMistral_MedCPTCE1024_85"]
    #model_names = ["BioMistral_SimCSE1024_BM25","BioMistral_SimCSE1024_25","BioMistral_SimCSE1024_50","BioMistral_SimCSE1024_75","BioMistral_SimCSE1024_85"]
    #model_names = ["BioMistral_Qwen1024_BM25","BioMistral_Qwen1024_25","BioMistral_Qwen1024_50","BioMistral_Qwen1024_75","BioMistral_Qwen1024_85"]
    #model_names = ["Openbio_MedCPT1024_BM25","Openbio_MedCPT1024_25","Openbio_MedCPT1024_50","Openbio_MedCPT1024_75","Openbio_MedCPT1024_85"]
    #model_names = ["Openbio_SimCSE1024_BM25","Openbio_SimCSE1024_25","Openbio_SimCSE1024_50","Openbio_SimCSE1024_75","Openbio_SimCSE1024_85"]
    #model_names = ["Openbio_Qwen1024_BM25","Openbio_Qwen1024_25","Openbio_Qwen1024_50","Openbio_Qwen1024_75","Openbio_Qwen1024_85"]
    #model_names = ["MedCPT1024_BM25_top3","MedCPT1024_25_top3","MedCPT1024_50_top3","MedCPT1024_75_top3","MedCPT1024_85_top3"]
    #model_names = ["medquad_qwen_1024_50"]
    #model_names = ["SimCSE1024_BM25_top3","SimCSE1024_25_top3","SimCSE1024_50_top3","SimCSE1024_75_top3","SimCSE1024_85_top3"]
    #model_names = ["LLama-8b_direct","ZS-LLama-8b","Qwen1024_50_top3"]
    model_names = ["Qwen1024_25", "Qwen512_25", "Qwen2k_25", "Rel_Qwen2k_25", "Rel_Qwen512_25", "Rel_Qwen1024_25" ]
    #model_names = [ "Rel_Qwen1024_50", "Rel_Qwen512_50", "Rel_Qwen2k_50",   ]
    #model_names = [ "Rel_BioMistral_Qwen1024_25", "Rel_BioMistral_Qwen1024_50", "Rel_MedCPTCE_1024_25", "Rel_MedCPTCE_1024_50",
     #             "Rel_Openbio_Qwen1024_25","Rel_Openbio_Qwen1024_50","Rel_SimCSE1024_25","Rel_SimCSE1024_50"
     #             ]
    
    
    # Create individual reports for each model
    evaluator = UnifiedMedicalQAEvaluator()
    comparison_results = []
    individual_reports = []
    
    for file_path, model_name in zip(file_paths, model_names):
        print(f"\nCreating detailed report for {model_name}...")
        
        # Run evaluation for this specific model
        results_df, metrics = evaluator.complete_evaluation_pipeline(
            file_path, 
            evaluation_types=['traditional', 'llm']
        )

        # Extract comparison metrics
        comparison_row = {'Model': model_name}
        key_metrics = [
            'avg_f1_score', 'avg_semantic_similarity', 'avg_rouge1_f1',
            'avg_answer_relevancy_score', 'avg_context_adherence_score',
            'retrieval_recall', 'mean_average_precision', 'accuracy',
             # Individual context performance  
            'Context1_hit_rate', 'Context2_hit_rate', 'Context3_hit_rate',
    
            # Context combinations
            'any_context_hit_rate', 'all_contexts_hit_rate', 'no_context_hit_rate',
            'only_Context1_hit_rate', 'only_Context2_hit_rate', 'only_Context3_hit_rate',
    
            # Pairwise redundancy
            'Context1_Context2_both_hit_rate', 'Context1_Context3_both_hit_rate', 
            'Context2_Context3_both_hit_rate',
    
            # Distribution patterns
            '0_contexts_hit_rate', '1_contexts_hit_rate', '2_contexts_hit_rate', '3_contexts_hit_rate'
        ]
        
        for metric in key_metrics:
            comparison_row[metric] = metrics.get(metric, None)
        
        comparison_results.append(comparison_row)
        
        # Create individual report
        report_path = evaluator.create_evaluation_report(
            results_df, 
            metrics, 
            f"report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        individual_reports.append(report_path)

    # Get comparison results
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv("model_comparison_BioMistral_MedCPTCE_1024.csv", index=False)
    
    # Create a combined comparison report
    create_combined_comparison_report(comparison_df, individual_reports)
    
    print(f"\nModel comparison complete!")
    print(f"- Comparison table: model_comparison.csv")
    print(f"- Individual reports: {', '.join(individual_reports)}")
    print(f"- Combined report: combined_comparison_report.txt")

