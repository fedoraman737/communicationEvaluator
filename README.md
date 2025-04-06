# Communication Evaluator

An LLM-powered system for evaluating communication skills based on responses to sample scenarios. The system analyzes responses and provides scores on key communication dimensions: Empathy, Positioning, and Persuasion.

## Features

- Process communication responses to predefined scenarios
- Evaluate communication effectiveness using LLM analysis (OpenAI or Anthropic)
- Score responses based on key communication domains:
  - **Empathy**: How well the response acknowledges and connects with emotions
  - **Positioning**: How effectively the response balances positive/negative sentiments with appropriate tone
  - **Persuasion**: Use of persuasion techniques like social proof and reciprocity
- Generate personalized feedback with specific strengths and areas for improvement
- Web-based interface for submitting and evaluating responses
- API endpoint for programmatic evaluation

## Setup

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys (see `.env.example`)
4. Run the application: `python run.py`

## Project Structure

- `app.py`: Main Flask application with routes and controllers
- `run.py`: Entry point script for running the application
- `evaluator/`: Core evaluation logic
  - `llm_evaluator.py`: LLM integration for response analysis
- `models/`: Data models
  - `evaluation.py`: Evaluation result data structure
  - `response.py`: User response data structure
  - `scenario.py`: Communication scenario data structure
- `static/`: CSS, JS, and other static files
- `templates/`: HTML templates for the web interface
  - `evaluation_result.html`: Results page showing scores and feedback

## API Usage

The system provides a REST API for programmatically evaluating communication responses:

```
POST /api/evaluate
{
    "scenario_id": "1",
    "advisor_id": "user123", 
    "response_text": "Your communication response text here"
}
```

## Configuration

Configuration is managed through environment variables (see `.env.example`):

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `LLM_PROVIDER`: Which LLM provider to use ('openai' or 'anthropic')
- `MODEL_NAME`: Specific model to use (e.g., 'gpt-4o' for OpenAI)

## Version History
- v1.2: Current development version
- v1.1: Correction to build errors
- v1.0: Initial release with basic evaluation functionality