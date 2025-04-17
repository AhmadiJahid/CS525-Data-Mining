import json
from MEDRagEvalFramework import MEDRagEvaluator

# Create a sample multiple-choice dataset
def create_sample_mc_dataset():
    """Create a sample multiple-choice dataset for testing"""
    sample_dataset = [
        {
            "question": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral",
            "options": {
                "A": "paralysis of the facial muscles.",
                "B": "paralysis of the facial muscles and loss of taste.",
                "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
                "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
            },
            "answer": "A"
        },
        {
            "question": "Which of the following is the first-line treatment for uncomplicated hypertension in diabetes?",
            "options": {
                "A": "Calcium channel blockers",
                "B": "Beta blockers",
                "C": "ACE inhibitors",
                "D": "Thiazide diuretics"
            },
            "answer": "C"
        },
        {
            "question": "The most common causative organism in community-acquired pneumonia is:",
            "options": {
                "A": "Klebsiella pneumoniae",
                "B": "Streptococcus pneumoniae",
                "C": "Mycoplasma pneumoniae",
                "D": "Haemophilus influenzae"
            },
            "answer": "B"
        }
    ]
    
    # Save the sample dataset
    with open("sample_medrag_mc_dataset.json", "w") as f:
        json.dump(sample_dataset, f, indent=2)
    
    return "sample_medrag_mc_dataset.json"

# Example RAG system output for multiple choice questions
def simulate_rag_mc_output():
    """Simulate outputs from a RAG system answering multiple choice questions"""
    
    # Sample retrieval results (optional for multiple choice)
    retrieval_results = [
        {
            "question_id": "q1",
            "retrieved_docs": ["doc45", "doc12", "doc78"]
        },
        {
            "question_id": "q2",
            "retrieved_docs": ["doc56", "doc23", "doc89"]
        },
        {
            "question_id": "q3",
            "retrieved_docs": ["doc34", "doc67", "doc92"]
        }
    ]
    
    # Sample multiple choice results
    mc_results = [
        {
            "question_id": "q1",
            "selected_option": "A"  # Correct
        },
        {
            "question_id": "q2",
            "selected_option": "B"  # Incorrect (correct is C)
        },
        {
            "question_id": "q3",
            "selected_option": "B"  # Correct
        }
    ]
    
    return retrieval_results, mc_results

# Main execution
if __name__ == "__main__":
    # Create sample dataset
    print("Creating sample multiple-choice dataset...")
    dataset_path = create_sample_mc_dataset()
    
    # Create evaluator with multiple_choice format
    print("Initializing MEDRag evaluator for multiple-choice format...")
    evaluator = MEDRagEvaluator(dataset_path=dataset_path, dataset_format="multiple_choice")
    
    # Simulate RAG output
    print("Simulating RAG system output...")
    retrieval_results, mc_results = simulate_rag_mc_output()
    
    # Evaluate
    print("Evaluating performance...")
    evaluation_results = evaluator.evaluate_all(
        retrieval_results=retrieval_results,
        multiple_choice_results=mc_results
    )
    
    # Print results
    print("\nMEDRag Multiple Choice Evaluation Results:")
    print(f"Overall Score: {evaluation_results['overall_score']:.4f}")
    
    if 'multiple_choice_metrics' in evaluation_results:
        mc_metrics = evaluation_results['multiple_choice_metrics']
        print(f"\nMultiple Choice Accuracy: {mc_metrics['accuracy']:.4f}")
        print(f"Correct Answers: {mc_metrics['correct_answers']}/{mc_metrics['total_questions']}")
    
    if 'retrieval_metrics' in evaluation_results:
        ret_metrics = evaluation_results['retrieval_metrics']
        print(f"\nRetrieval Score: {ret_metrics.get('retrieval_score', 0):.4f}")
        
 