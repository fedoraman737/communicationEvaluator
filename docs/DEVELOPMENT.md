# Development Guide

This guide provides technical information for developers working on the Communication Evaluator project.

## Architecture Overview

The application follows a typical Flask web application architecture:

```
communication_evaluator/
├── app.py              # Main Flask application
├── run.py              # Entry point script
├── evaluator/          # Core evaluation logic
│   └── llm_evaluator.py # LLM integration
├── models/             # Data models
│   ├── evaluation.py   # Evaluation results
│   ├── response.py     # User responses
│   └── scenario.py     # Communication scenarios
├── static/             # Static assets
├── templates/          # HTML templates
└── tests/              # Test cases
```

## Core Components

### Models

The application uses simple dataclass models:

- `Scenario`: Represents a communication scenario to respond to
- `Response`: Represents a user's response to a scenario
- `Evaluation`: Stores the evaluation results including scores and feedback

### Evaluator

The `LLMEvaluator` class in `evaluator/llm_evaluator.py` handles the integration with LLM providers (OpenAI and Anthropic). It:

1. Constructs evaluation prompts
2. Submits prompts to LLM APIs
3. Processes and formats evaluation results

### Web Interface

Flask routes in `app.py` handle:
- Displaying scenarios
- Accepting user responses
- Showing evaluation results

## Class Diagrams

### Model Classes

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Scenario   │      │  Response   │      │ Evaluation  │
├─────────────┤      ├─────────────┤      ├─────────────┤
│ id          │      │ id          │      │ id          │
│ title       │      │ advisor_id  │      │ empathy_score│
│ description │      │ scenario_id │      │ position_score│
│ category    │      │ text        │      │ persuasion_score│
└─────────────┘      │ submitted_at│      │ overall_score│
                     └─────────────┘      │ strengths    │
                                          │ areas_for_imp│
                                          │ feedback     │
                                          │ response_id  │
                                          │ created_at   │
                                          └─────────────┘
```

## LLM Integration

The application supports multiple LLM providers through a common interface. The provider is selected through the `LLM_PROVIDER` environment variable.

### Adding a New LLM Provider

To add support for a new LLM provider:

1. Add client initialization in the `LLMEvaluator.__init__` method
2. Create a new `_evaluate_with_provider` method
3. Update the `evaluate_response` method to use your new provider

Example:

```python
def _evaluate_with_new_provider(self, prompt: str) -> Dict[str, Any]:
    """
    Evaluate a response using the new provider's API.
    
    Args:
        prompt: The evaluation prompt
        
    Returns:
        A dictionary containing the evaluation results
    """
    try:
        # Implement API call to new provider
        response = self.client.some_method(
            prompt=prompt,
            # Additional parameters
        )
        
        # Parse and return results
        evaluation_result = json.loads(response.result)
        return evaluation_result
        
    except Exception as e:
        logger.error(f"Error with new provider evaluation: {str(e)}")
        raise
```

## Prompt Engineering

The evaluation prompt is crucial for effective evaluations. The current prompt defines:

1. The role of the LLM as a communication expert
2. The specific evaluation dimensions (empathy, positioning, persuasion)
3. The expected output format

To modify the evaluation criteria, edit the `_create_evaluation_prompt` method in `llm_evaluator.py`.

## API Extension

The REST API can be extended by adding new endpoints to `app.py`. For example, to add an endpoint that retrieves historical evaluations:

```python
@app.route('/api/evaluations/<advisor_id>', methods=['GET'])
def get_evaluations(advisor_id):
    # In a real application, fetch from database
    # For demo, return mock data
    
    return jsonify({
        'evaluations': [
            # Evaluation objects
        ]
    })
```

## Testing

The project uses pytest for testing. Key test areas include:

1. Model validations
2. Evaluator logic (including mock LLM responses)
3. API endpoints
4. Template rendering

### Example Test

```python
def test_evaluation_model():
    """Test that the Evaluation model is created correctly."""
    eval = Evaluation(
        empathy_score=8.5,
        positioning_score=7.0,
        persuasion_score=6.5,
        overall_score=7.3,
        strengths=["Good empathy", "Clear explanation"],
        areas_for_improvement=["Improve tone"],
        feedback="Good job overall"
    )
    
    assert eval.id is not None
    assert eval.empathy_score == 8.5
    assert len(eval.strengths) == 2
```

## Future Development

Areas for further development:

1. **Database Integration**: Replace in-memory data with persistent storage
2. **User Authentication**: Add user accounts and login
3. **Historical Tracking**: Store and display evaluation history
4. **Custom Scoring Models**: Allow different scoring rubrics for different scenarios
5. **Enhanced Visualization**: Add charts and graphs for evaluation results 