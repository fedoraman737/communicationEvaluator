# Communication Evaluator

An LLM-powered system for evaluating advisor communication skills based on survey responses.

## Features

- Process survey responses containing advisor communications
- Evaluate communication effectiveness using LLM analysis
- Score responses based on key communication domains (Empathy, Positioning, Persuasion)
- Generate personalized feedback for improvement

## Setup

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys (see `.env.example`)
4. Run the application: `python app.py`

## Project Structure

- `app.py`: Main application entry point
- `evaluator/`: Core evaluation logic
- `models/`: Data models
- `static/`: CSS, JS, and other static files
- `templates/`: HTML templates for the web interface