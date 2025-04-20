import json
from MEDRagEvalFramework import MEDRagEvaluator

# Sample MEDRag dataset structure
def create_sample_dataset():
    """Create a small sample MEDRag dataset for testing"""
    sample_dataset = {
        "metadata": {
            "version": "1.0",
            "name": "MEDRag Sample Dataset",
            "description": "A small sample dataset for testing MEDRag evaluation"
        },
        "questions": {
            "q1": {
                "text": "What is the first-line treatment for uncomplicated type 2 diabetes?",
                "question_type": "treatment",
                "clinical_context": "Adult patient newly diagnosed with type 2 diabetes, HbA1c 7.5%",
                "relevant_docs": ["doc1", "doc5", "doc8"],
                "reference_answer": "Metformin is the first-line pharmacologic agent for type 2 diabetes. It should be initiated at the time of diagnosis unless contraindicated. Metformin is effective, safe, and inexpensive, and may reduce risk of cardiovascular events and death.",
                "key_concepts": ["metformin", "first-line", "type 2 diabetes", "contraindicated", "cardiovascular"],
                "expected_citations": [
                    {"doc_id": "doc1", "text": "Metformin is the preferred initial pharmacologic agent for type 2 diabetes."},
                    {"doc_id": "doc5", "text": "Metformin may reduce risk of cardiovascular events and death."}
                ]
            },
            "q2": {
                "text": "What are the diagnostic criteria for multiple sclerosis?",
                "question_type": "diagnosis",
                "clinical_context": "30-year-old female with episodic neurological symptoms",
                "relevant_docs": ["doc3", "doc7", "doc12"],
                "reference_answer": "Multiple sclerosis is diagnosed based on demonstrating dissemination of lesions in space and time using clinical and MRI findings, with the exclusion of alternative diagnoses. The McDonald criteria are commonly used, requiring evidence of damage in at least two separate areas of the central nervous system and evidence that the damage occurred at different time points.",
                "key_concepts": ["dissemination in space", "dissemination in time", "McDonald criteria", "MRI", "exclusion of alternatives"],
                "expected_citations": [
                    {"doc_id": "doc3", "text": "The 2017 McDonald criteria for MS diagnosis require demonstration of dissemination of lesions in space and time."},
                    {"doc_id": "doc7", "text": "MRI findings are central to the diagnosis of multiple sclerosis."}
                ]
            }
        }
    }
    
    # Save the sample dataset
    with open("sample_medrag_dataset.json", "w") as f:
        json.dump(sample_dataset, f, indent=2)
    
    return "sample_medrag_dataset.json"

# Sample RAG system output
def create_sample_rag_outputs():
    """Create sample outputs from a RAG system for evaluation"""
    
    # Sample retrieval results
    retrieval_results = [
        {
            "question_id": "q1",
            "retrieved_docs": ["doc1", "doc5", "doc9", "doc11", "doc2"]
        },
        {
            "question_id": "q2",
            "retrieved_docs": ["doc3", "doc7", "doc4", "doc8", "doc12"]
        }
    ]
    
    # Sample generation results
    generation_results = [
        {
            "question_id": "q1",
            "generated_answer": "Metformin is the first-line treatment for uncomplicated type 2 diabetes. It is effective, relatively safe, and inexpensive. Treatment should be initiated at diagnosis unless there are contraindications such as severe renal impairment. Metformin works by decreasing hepatic glucose production and increasing insulin sensitivity. Some studies suggest it may reduce cardiovascular events.",
            "citations": [
                {"text": "Metformin is the preferred initial pharmacologic agent for type 2 diabetes.", "doc_id": "doc1"},
                {"text": "Metformin works by decreasing hepatic glucose production.", "doc_id": "doc2"},
                {"text": "Metformin may reduce risk of cardiovascular events.", "doc_id": "doc5"}
            ]
        },
        {
            "question_id": "q2",
            "generated_answer": "Multiple sclerosis is diagnosed using the McDonald criteria, which requires evidence of CNS lesions disseminated in space (multiple locations) and time (occurring at different times). MRI is the main tool for diagnosis, showing characteristic lesions. Cerebrospinal fluid analysis may show oligoclonal bands. The diagnosis also requires exclusion of alternative conditions that might better explain the symptoms.",
            "citations": [
                {"text": "The 2017 McDonald criteria for MS diagnosis require demonstration of dissemination of lesions in space and time.", "doc_id": "doc3"},
                {"text": "MRI findings are central to the diagnosis of multiple sclerosis.", "doc_id": "doc7"},
                {"text": "CSF analysis might show oligoclonal bands in MS patients.", "doc_id": "doc4"}
            ]
        }
    ]
    
    return retrieval_results, generation_results

# Main execution
if __name__ == "__main__":
    # Create sample dataset and RAG outputs
    dataset_path = create_sample_dataset()
    retrieval_results, generation_results = create_sample_rag_outputs()
    
    # Initialize the evaluator
    evaluator = MEDRagEvaluator(dataset_path=dataset_path)
    
    # Run comprehensive evaluation
    print("Running MEDRag evaluation...")
    eval_results = evaluator.evaluate_all(retrieval_results, generation_results)
    
    # Output detailed report
    report = evaluator.output_detailed_report(eval_results, "medrag_evaluation_results.json")
    
    # Display summary
    print("\nMEDRag Evaluation Summary:")
    print(f"Overall Score: {report['overall_score']}")
    print("\nComponent Scores:")
    for component, score in report['component_scores'].items():
        print(f"- {component}: {score}")
    
    print("\nDetailed metrics are available in the generated report file: medrag_evaluation_results.json")