# Eval Protocol

## Two-Phase Evaluation

Each eval implements two functions for batch processing:

### Phase 1: `prepare_requests(model_name: str, **kwargs) -> Dict`

Returns:
```python
{
    "requests": [
        {
            # For chat completions:
            "messages": [...],
            "max_tokens": 512,
            "temperature": 0.0,
            
            # OR for text completions:
            "prompt": "...",
            "max_tokens": 512,
            "temperature": 0.0,
        },
        ...
    ],
    "metadata": {...}   # Any data needed for scoring
}
```

### Phase 2: `score_results(requests, responses, metadata) -> Dict`

Returns:
```python
{
    "metrics": {...},   # Evaluation metrics
    "details": {...}    # Optional detailed results
}
```

## Adding a New Eval

1. Create `evals/your_eval/eval.py` with the two functions
2. Add eval name to `config.json` under `default_evals`
3. That's it!

## Running

```bash
# List evals to run
python run_eval.py list

# Get requests for an eval
python run_eval.py prepare ifeval-da my-model

# Score results
echo '{"requests": [...], "responses": [...], "metadata": {...}}' | \
    python run_eval.py score ifeval-da
```