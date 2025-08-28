"""
Simple example evaluation to test basic model capabilities.
"""

from typing import Dict, List


def prepare_requests(model_name: str, **kwargs) -> Dict:
    """
    Prepare simple test requests.
    """
    prompts = [
        "Translate to French: Hello, how are you?",
        "What is 2 + 2?",
        "Complete this sentence: The capital of France is",
    ]
    
    # Allow limiting prompts for testing
    max_prompts = kwargs.get("max_prompts", len(prompts))
    prompts = prompts[:max_prompts]
    
    requests = []
    expected_answers = []
    
    for prompt in prompts:
        requests.append({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.0,
        })
        
        # Store expected answers for basic checking
        if "French" in prompt:
            expected_answers.append("bonjour")
        elif "2 + 2" in prompt:
            expected_answers.append("4")
        elif "France" in prompt:
            expected_answers.append("paris")
    
    return {
        "requests": requests,
        "metadata": {
            "prompts": prompts,
            "expected_answers": expected_answers,
        }
    }


def score_results(requests: List[Dict], responses: List[Dict], metadata: Dict) -> Dict:
    """
    Score the responses - just check if responses were generated successfully.
    """
    prompts = metadata.get("prompts", [])
    expected = metadata.get("expected_answers", [])
    
    successful = 0
    contains_expected = 0
    errors = 0
    
    for i, response in enumerate(responses):
        if "error" in response:
            errors += 1
            continue
            
        try:
            content = response["choices"][0]["message"]["content"]
            successful += 1
            
            # Basic check if expected answer is in response
            if i < len(expected) and expected[i].lower() in content.lower():
                contains_expected += 1
                
        except (KeyError, IndexError):
            errors += 1
    
    return {
        "metrics": {
            "success_rate": successful / len(responses) if responses else 0,
            "contains_expected_rate": contains_expected / len(responses) if responses else 0,
            "n_samples": len(responses),
            "n_successful": successful,
            "n_errors": errors,
        },
        "details": {
            "prompts": prompts[:5],  # Sample for inspection
        }
    }