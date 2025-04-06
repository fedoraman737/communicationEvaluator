# Installation Guide

This guide provides detailed instructions for setting up the Communication Evaluator application.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tool (recommended, e.g., venv, conda)
- OpenAI API key and/or Anthropic API key

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/communicationEvaluator.git
cd communicationEvaluator
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv (Python's built-in virtual environment)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys:

```
# LLM API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Application Settings
FLASK_ENV=development
LLM_PROVIDER=openai  # Options: openai, anthropic
MODEL_NAME=gpt-4o  # For OpenAI: gpt-4o, gpt-3.5-turbo; For Anthropic: claude-3-sonnet-20240229
```

### 5. Run the Application

```bash
python run.py
```

The application will be available at http://127.0.0.1:5000

## Troubleshooting

### API Key Issues

If you encounter errors related to API keys:

1. Verify that your API keys are correctly set in the `.env` file
2. Ensure your API keys have sufficient credits/quota
3. Check that you've activated any required API access in your provider's dashboard

### Dependency Issues

If you encounter dependency-related errors:

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Port Already in Use

If port 5000 is already in use, you can modify the port in `run.py`:

```python
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change to an available port
``` 