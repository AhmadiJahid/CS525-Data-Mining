import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, ndcg_score, accuracy_score
from typing import List, Dict, Any, Tuple, Set, Optional, Union
import re
from collections import defaultdict

class QADataset:
    """
    A dataset class for multiple-choice question answering tasks.
    Implementation similar to what's used in MEDRag.
    """
    
    def __init__(self, dataset_name: str, data_path: Optional[str] = None):
        """
        Initialize the QA dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., "mmlu", "medqa")
            data_path: Optional path to the dataset file. If None, uses default path.
        """
        self.dataset_name = dataset_name
        
        # Load the dataset from file
        if data_path:
            self.data = self._load_data(data_path)
        else:
            # In real implementation, this would look for default paths
            self.data = []
            
    def _load_data(self, path: str) -> List[Dict]:
        """Load the dataset from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    def __len__(self) -> int:
        """Return the number of questions in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a specific question by index."""
        return self.data[idx]


class MEDRagEvaluator:
    """
    A comprehensive evaluation framework for Medical RAG systems based on the MEDRag benchmark.
    Supports both multiple-choice QA and free-text generation evaluation.
    """
    
    def __init__(self, dataset_path: str, corpus_path: str = None, dataset_format: str = "generation"):
        """
        Initialize the MEDRag evaluator.
        
        Args:
            dataset_path: Path to the MEDRag dataset JSON file
            corpus_path: Path to the corpus documents (optional if your RAG system has its own corpus)
            dataset_format: Format of the dataset - "generation" (default) or "multiple_choice"
        """
        self.dataset_format = dataset_format
        
        if dataset_format == "multiple_choice":
            # Load as QADataset for multiple choice
            dataset_name = dataset_path.split("/")[-1].split(".")[0]
            self.qa_dataset = QADataset(dataset_name, dataset_path)
            self.dataset = self._convert_qa_to_internal_format()
        else:
            # Load as regular dataset for generation
            self.dataset = self._load_dataset(dataset_path)
            
        self.corpus = self._load_corpus(corpus_path) if corpus_path else None
        
        # Define metrics weights
        self.weights = {
            "retrieval": 0.4,
            "factual_correctness": 0.3,
            "citation_accuracy": 0.2,
            "clinical_relevance": 0.1
        }
    
    def _convert_qa_to_internal_format(self) -> Dict:
        """Convert QADataset format to internal format for evaluation."""
        internal_format = {
            "metadata": {
                "version": "1.0",
                "name": f"MEDRag {self.qa_dataset.dataset_name} Dataset",
                "description": "Multiple choice medical QA dataset"
            },
            "questions": {}
        }
        
        for idx, qa_item in enumerate(self.qa_dataset.data):
            question_id = f"q{idx+1}"
            
            # Extract options text as a list
            options_list = [value for key, value in qa_item["options"].items()]
            
            # Correct answer text
            correct_answer_key = qa_item["answer"]
            correct_answer_text = qa_item["options"][correct_answer_key]
            
            internal_format["questions"][question_id] = {
                "text": qa_item["question"],
                "question_type": "multiple_choice",
                "options": qa_item["options"],
                "answer_key": correct_answer_key,
                "reference_answer": correct_answer_text,
                # We don't have these for multiple choice, but including empty versions for compatibility
                "relevant_docs": [],
                "key_concepts": [],
                "expected_citations": []
            }
            
        return internal_format
    
    def _load_dataset(self, path: str) -> Dict:
        """Load the MEDRag dataset from a JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_corpus(self, path: str) -> Dict:
        """Load the corpus documents."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def evaluate_retrieval(self, 
                          retrieval_results: List[Dict], 
                          top_k: int = 10) -> Dict[str, float]:
        """
        Evaluate the retrieval component of the RAG system.
        
        Args:
            retrieval_results: List of dictionaries with keys:
                - 'question_id': ID of the question
                - 'retrieved_docs': List of retrieved document IDs
            top_k: Number of top documents to consider
            
        Returns:
            Dictionary of retrieval metrics
        """
        precision_list = []
        recall_list = []
        ndcg_list = []
        
        for result in retrieval_results:
            question_id = result['question_id']
            retrieved_docs = result['retrieved_docs'][:top_k]
            
            # Get relevant documents from the dataset
            relevant_docs = self.dataset['questions'][question_id]['relevant_docs']
            
            # Calculate precision and recall
            retrieved_set = set(retrieved_docs)
            relevant_set = set(relevant_docs)
            
            retrieved_relevant = retrieved_set.intersection(relevant_set)
            
            precision = len(retrieved_relevant) / len(retrieved_set) if retrieved_set else 0
            recall = len(retrieved_relevant) / len(relevant_set) if relevant_set else 0
            
            precision_list.append(precision)
            recall_list.append(recall)
            
            # Calculate NDCG
            y_true = np.zeros(len(retrieved_docs))
            for i, doc_id in enumerate(retrieved_docs):
                if doc_id in relevant_set:
                    y_true[i] = 1
                    
            y_score = np.array([1.0 / (i + 1) for i in range(len(retrieved_docs))])
            
            try:
                ndcg = ndcg_score(np.array([y_true]), np.array([y_score]))
                ndcg_list.append(ndcg)
            except:
                # Handle edge cases where NDCG calculation fails
                pass
        
        # Calculate mean metrics
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "ndcg": avg_ndcg,
            "retrieval_score": (avg_precision + avg_recall + avg_f1 + avg_ndcg) / 4
        }
    
    def evaluate_factual_correctness(self, 
                                   generation_results: List[Dict]) -> Dict[str, float]:
        """
        Evaluate the factual correctness of generated answers.
        
        Args:
            generation_results: List of dictionaries with keys:
                - 'question_id': ID of the question
                - 'generated_answer': Generated answer text
                
        Returns:
            Dictionary of factual correctness metrics
        """
        correctness_scores = []
        completeness_scores = []
        
        for result in generation_results:
            question_id = result['question_id']
            generated_answer = result['generated_answer']
            
            # Get reference answer
            reference_answer = self.dataset['questions'][question_id]['reference_answer']
            
            # Get key medical concepts that should be in the answer
            key_concepts = self.dataset['questions'][question_id].get('key_concepts', [])
            
            # Check how many key concepts are covered
            concepts_covered = 0
            for concept in key_concepts:
                if concept.lower() in generated_answer.lower():
                    concepts_covered += 1
            
            completeness = concepts_covered / len(key_concepts) if key_concepts else 0
            
            # Check for factual contradictions - simplified approach
            # In a real implementation, you might use NLI models or more sophisticated methods
            contradictions = self._detect_contradictions(generated_answer, reference_answer)
            correctness = 1.0 - (contradictions / 10)  # Normalize to 0-1 scale
            
            correctness_scores.append(max(0, correctness))
            completeness_scores.append(completeness)
        
        return {
            "factual_correctness": np.mean(correctness_scores),
            "answer_completeness": np.mean(completeness_scores),
            "factual_score": (np.mean(correctness_scores) + np.mean(completeness_scores)) / 2
        }
    
    def _detect_contradictions(self, generated: str, reference: str) -> int:
        """
        Simple method to detect potential contradictions.
        In practice, you would use a more sophisticated approach.
        
        Returns:
            Estimated number of contradictions (0-10 scale)
        """
        # This is a placeholder - in reality, you'd use:
        # 1. Medical NLI models
        # 2. Specialized medical contradiction detection
        # 3. Knowledge graph verification
        
        # Simple negative phrase detection
        contradictions = 0
        
        # Check for opposing medical terms
        opposing_terms = [
            ("increase", "decrease"),
            ("high", "low"),
            ("positive", "negative"),
            ("present", "absent"),
            ("recommended", "contraindicated")
        ]
        
        for term1, term2 in opposing_terms:
            if term1 in generated.lower() and term2 in reference.lower():
                contradictions += 1
            if term2 in generated.lower() and term1 in reference.lower():
                contradictions += 1
                
        return min(contradictions, 10)
    
    def evaluate_citation_accuracy(self, 
                                 generation_results: List[Dict]) -> Dict[str, float]:
        """
        Evaluate the accuracy of citations in generated answers.
        
        Args:
            generation_results: List of dictionaries with keys:
                - 'question_id': ID of the question
                - 'generated_answer': Generated answer text
                - 'citations': List of citation dictionaries with:
                    - 'text': Cited text
                    - 'doc_id': Document ID
                
        Returns:
            Dictionary of citation accuracy metrics
        """
        citation_presence_scores = []
        citation_relevance_scores = []
        citation_coverage_scores = []
        
        for result in generation_results:
            question_id = result['question_id']
            generated_answer = result['generated_answer']
            citations = result.get('citations', [])
            
            # Get expected citations
            expected_citations = self.dataset['questions'][question_id].get('expected_citations', [])
            
            # Check citation presence
            citation_presence = len(citations) > 0
            citation_presence_scores.append(1.0 if citation_presence else 0.0)
            
            # Calculate citation relevance
            if not citations:
                citation_relevance_scores.append(0.0)
                citation_coverage_scores.append(0.0)
                continue
                
            # Check if cited documents are in the expected set
            relevant_citations = 0
            for citation in citations:
                doc_id = citation['doc_id']
                if doc_id in [cite['doc_id'] for cite in expected_citations]:
                    relevant_citations += 1
            
            relevance_score = relevant_citations / len(citations) if citations else 0
            citation_relevance_scores.append(relevance_score)
            
            # Calculate citation coverage
            cited_concepts = set()
            for citation in citations:
                citation_text = citation['text'].lower()
                for concept in self.dataset['questions'][question_id].get('key_concepts', []):
                    if concept.lower() in citation_text:
                        cited_concepts.add(concept.lower())
            
            coverage = len(cited_concepts) / len(self.dataset['questions'][question_id].get('key_concepts', [])) if self.dataset['questions'][question_id].get('key_concepts', []) else 0
            citation_coverage_scores.append(coverage)
        
        return {
            "citation_presence": np.mean(citation_presence_scores),
            "citation_relevance": np.mean(citation_relevance_scores),
            "citation_coverage": np.mean(citation_coverage_scores),
            "citation_score": (np.mean(citation_presence_scores) + np.mean(citation_relevance_scores) + np.mean(citation_coverage_scores)) / 3
        }
    
    def evaluate_clinical_relevance(self, 
                                  generation_results: List[Dict]) -> Dict[str, float]:
        """
        Evaluate the clinical relevance of generated answers.
        
        Args:
            generation_results: List of dictionaries with keys:
                - 'question_id': ID of the question
                - 'generated_answer': Generated answer text
                
        Returns:
            Dictionary of clinical relevance metrics
        """
        # In practice, this would require medical expert evaluation
        # Here, we'll provide a simple implementation based on keyword matching
        
        clinical_relevance_scores = []
        clinical_safety_scores = []
        
        for result in generation_results:
            question_id = result['question_id']
            generated_answer = result['generated_answer']
            
            # Get clinical context and question type
            question_type = self.dataset['questions'][question_id].get('question_type', '')
            clinical_context = self.dataset['questions'][question_id].get('clinical_context', '')
            
            # Calculate relevance score
            relevance_score = self._calculate_clinical_relevance(generated_answer, question_type, clinical_context)
            clinical_relevance_scores.append(relevance_score)
            
            # Check for safety issues
            safety_score = self._check_clinical_safety(generated_answer)
            clinical_safety_scores.append(safety_score)
        
        return {
            "clinical_relevance": np.mean(clinical_relevance_scores),
            "clinical_safety": np.mean(clinical_safety_scores),
            "clinical_score": (np.mean(clinical_relevance_scores) + np.mean(clinical_safety_scores)) / 2
        }
    
    def _calculate_clinical_relevance(self, answer: str, question_type: str, clinical_context: str) -> float:
        """
        Calculate the clinical relevance of an answer.
        
        Returns:
            Clinical relevance score (0 to 1)
        """
        # This is a simplified approach - in practice, you would use:
        # 1. Medical expert evaluation
        # 2. Clinical guidelines compliance check
        # 3. Relevance to the specific clinical scenario
        
        answer = answer.lower()
        
        # Check if the answer addresses the question type
        question_type_relevance = 0.0
        question_types = {
            "diagnosis": ["diagnosis", "differential diagnosis", "diagnostic", "symptoms", "signs", "clinical presentation"],
            "treatment": ["treatment", "therapy", "medication", "therapeutic", "management", "intervention"],
            "prognosis": ["prognosis", "outcome", "survival", "mortality", "complication", "long-term"],
            "epidemiology": ["prevalence", "incidence", "population", "demographic", "risk factor", "epidemiology"]
        }
        
        if question_type in question_types:
            keywords = question_types[question_type]
            for keyword in keywords:
                if keyword in answer:
                    question_type_relevance = 1.0
                    break
                    
        # Check if the answer addresses the clinical context
        context_relevance = 0.0
        if clinical_context:
            context_terms = clinical_context.lower().split()
            context_matches = sum(1 for term in context_terms if term in answer and len(term) > 3)
            context_relevance = min(1.0, context_matches / 5)  # Normalize to 0-1
        
        return (question_type_relevance + context_relevance) / 2
    
    def _check_clinical_safety(self, answer: str) -> float:
        """
        Check if the answer has potential clinical safety issues.
        
        Returns:
            Safety score (0 to 1, where 1 is safe)
        """
        # This is a simplified approach - in practice, you would:
        # 1. Use medical experts to evaluate safety
        # 2. Check against established guidelines
        # 3. Check for dangerous recommendations
        
        answer = answer.lower()
        
        # Check for safety phrases
        safety_issues = 0
        unsafe_patterns = [
            r"always (.*?) without consulting",
            r"never seek medical attention",
            r"ignore (.*?) symptoms",
            r"stop (.*?) medication",
            r"alternative to (.*?) vaccine",
            r"replace (.*?) treatment"
        ]
        
        uncertainty_markers = [
            "may", "might", "could", "suggest", "consider", "possible", "potential",
            "consult", "physician", "healthcare provider", "doctor"
        ]
        
        # Check for unsafe patterns
        for pattern in unsafe_patterns:
            if re.search(pattern, answer):
                safety_issues += 1
        
        # Check for appropriate uncertainty
        uncertainty_score = 0
        for marker in uncertainty_markers:
            if marker in answer:
                uncertainty_score += 1
        uncertainty_present = min(1.0, uncertainty_score / 3)
        
        # Calculate final safety score
        safety_score = 1.0 - (safety_issues / 10) + uncertainty_present
        return min(1.0, max(0.0, safety_score))
    
    def evaluate_multiple_choice(self, results: List[Dict]) -> Dict[str, float]:
        """
        Evaluate multiple-choice answers.
        
        Args:
            results: List of dictionaries with keys:
                - 'question_id': ID of the question
                - 'selected_option': The selected option key (e.g., 'A', 'B', 'C', 'D')
                - 'retrieved_docs': (Optional) List of retrieved document IDs
                
        Returns:
            Dictionary of evaluation metrics
        """
        accuracy = 0
        question_count = 0
        
        for result in results:
            question_id = result['question_id']
            selected_option = result.get('selected_option')
            
            if not selected_option or question_id not in self.dataset['questions']:
                continue
                
            correct_option = self.dataset['questions'][question_id].get('answer_key')
            
            if selected_option == correct_option:
                accuracy += 1
                
            question_count += 1
            
        accuracy_score = accuracy / question_count if question_count > 0 else 0
        
        return {
            "accuracy": accuracy_score,
            "total_questions": question_count,
            "correct_answers": accuracy
        }
    
    def evaluate_all(self, 
                    retrieval_results: List[Dict] = None, 
                    generation_results: List[Dict] = None,
                    multiple_choice_results: List[Dict] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on retrieval and generation results.
        
        Args:
            retrieval_results: List of retrieval result dictionaries (optional)
            generation_results: List of generation result dictionaries (optional)
            multiple_choice_results: List of multiple choice result dictionaries (optional)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        if self.dataset_format == "multiple_choice" and multiple_choice_results:
            # For multiple choice format
            mc_metrics = self.evaluate_multiple_choice(multiple_choice_results)
            
            # Check if we also have retrieval results to evaluate
            if retrieval_results:
                retrieval_metrics = self.evaluate_retrieval(retrieval_results)
                
                return {
                    "retrieval_metrics": retrieval_metrics,
                    "multiple_choice_metrics": mc_metrics,
                    "overall_score": (0.3 * retrieval_metrics["retrieval_score"] + 0.7 * mc_metrics["accuracy"])
                }
            else:
                return {
                    "multiple_choice_metrics": mc_metrics,
                    "overall_score": mc_metrics["accuracy"]
                }
        else:
            # For generation format
            retrieval_metrics = self.evaluate_retrieval(retrieval_results) if retrieval_results else {"retrieval_score": 0}
            factual_metrics = self.evaluate_factual_correctness(generation_results) if generation_results else {"factual_score": 0}
            citation_metrics = self.evaluate_citation_accuracy(generation_results) if generation_results else {"citation_score": 0}
            clinical_metrics = self.evaluate_clinical_relevance(generation_results) if generation_results else {"clinical_score": 0}
            
            # Calculate weighted overall score
            overall_score = (
                self.weights["retrieval"] * retrieval_metrics["retrieval_score"] +
                self.weights["factual_correctness"] * factual_metrics["factual_score"] +
                self.weights["citation_accuracy"] * citation_metrics["citation_score"] +
                self.weights["clinical_relevance"] * clinical_metrics["clinical_score"]
            )
            
            return {
                "retrieval_metrics": retrieval_metrics,
                "factual_metrics": factual_metrics,
                "citation_metrics": citation_metrics,
                "clinical_metrics": clinical_metrics,
                "overall_score": overall_score
            }
    
    def output_detailed_report(self, evaluation_results: Dict, output_path: str = "medrag_report.json"):
        """
        Generate a detailed report of the evaluation results.
        
        Args:
            evaluation_results: Results from evaluate_all method
            output_path: Path to save the report
        """
        # Format the detailed report with human-readable scores
        report = {
            "overall_score": f"{evaluation_results['overall_score']*100:.2f}%",
            "component_scores": {
                "retrieval": f"{evaluation_results['retrieval_metrics']['retrieval_score']*100:.2f}%",
                "factual_correctness": f"{evaluation_results['factual_metrics']['factual_score']*100:.2f}%",
                "citation_accuracy": f"{evaluation_results['citation_metrics']['citation_score']*100:.2f}%",
                "clinical_relevance": f"{evaluation_results['clinical_metrics']['clinical_score']*100:.2f}%"
            },
            "detailed_metrics": {
                "retrieval": {
                    "precision": f"{evaluation_results['retrieval_metrics']['precision']*100:.2f}%",
                    "recall": f"{evaluation_results['retrieval_metrics']['recall']*100:.2f}%",
                    "f1": f"{evaluation_results['retrieval_metrics']['f1']*100:.2f}%",
                    "ndcg": f"{evaluation_results['retrieval_metrics']['ndcg']*100:.2f}%"
                },
                "factual_correctness": {
                    "correctness": f"{evaluation_results['factual_metrics']['factual_correctness']*100:.2f}%",
                    "completeness": f"{evaluation_results['factual_metrics']['answer_completeness']*100:.2f}%"
                },
                "citation": {
                    "presence": f"{evaluation_results['citation_metrics']['citation_presence']*100:.2f}%",
                    "relevance": f"{evaluation_results['citation_metrics']['citation_relevance']*100:.2f}%",
                    "coverage": f"{evaluation_results['citation_metrics']['citation_coverage']*100:.2f}%"
                },
                "clinical": {
                    "relevance": f"{evaluation_results['clinical_metrics']['clinical_relevance']*100:.2f}%",
                    "safety": f"{evaluation_results['clinical_metrics']['clinical_safety']*100:.2f}%"
                }
            }
        }
        
        # Save the report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


# Example usage
if __name__ == "__main__":
    # Initialize the evaluator
    evaluator = MEDRagEvaluator(dataset_path="path/to/medrag_dataset.json")
    
    # Example retrieval results
    retrieval_results = [
        {
            "question_id": "q1",
            "retrieved_docs": ["doc1", "doc2", "doc3"]
        },
        # More retrieval results...
    ]
    
    # Example generation results
    generation_results = [
        {
            "question_id": "q1",
            "generated_answer": "The treatment for condition X includes medication Y at a dose of Z mg daily.",
            "citations": [
                {"text": "Medication Y is effective for condition X", "doc_id": "doc1"},
                {"text": "The recommended dose is Z mg daily", "doc_id": "doc2"}
            ]
        },
        # More generation results...
    ]
    
    # Run evaluation
    evaluation_results = evaluator.evaluate_all(retrieval_results, generation_results)
    
    # Generate report
    report = evaluator.output_detailed_report(evaluation_results)
    print(f"Overall MEDRag score: {report['overall_score']}")