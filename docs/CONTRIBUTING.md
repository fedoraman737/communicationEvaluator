# Contributing to Communication Evaluator

Thank you for considering contributing to Communication Evaluator! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. Harassment or abusive behavior will not be tolerated.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

1. Clear, descriptive title
2. Steps to reproduce the bug
3. Expected behavior
4. Actual behavior
5. Screenshots (if applicable)
6. Environment details (OS, browser, etc.)

### Suggesting Features

We welcome feature suggestions! To suggest a feature:

1. Create an issue with a clear, descriptive title
2. Describe the feature and its benefits
3. Provide any relevant examples or mockups

### Development Workflow

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Write tests for your changes (if applicable)
5. Run existing tests to ensure your changes don't break anything
6. Submit a pull request

### Branch Naming Convention

Use the following format for branch names:

- For features: `feature/short-description`
- For bug fixes: `fix/short-description`
- For documentation: `docs/short-description`

### Commit Message Guidelines

Write clear, concise commit messages that explain what your changes do and why. Use present tense and imperative mood (e.g., "Add feature" not "Added feature").

## Setting Up the Development Environment

1. Clone your fork of the repository
2. Create a virtual environment
3. Install dependencies with `pip install -r requirements.txt`
4. Set up environment variables according to `.env.example`
5. Run the application with `python run.py`

## Testing

Run tests with pytest:

```bash
pytest
```

Please ensure all tests pass before submitting a pull request. If you're adding new functionality, please include tests for it.

## Code Style

We follow PEP 8 style guidelines for Python code. You can use tools like flake8 or black to ensure your code meets these standards:

```bash
# Install formatting tools
pip install flake8 black

# Check code style
flake8 .

# Format code
black .
```

## Documentation

Please update documentation when changing code:

- Update docstrings for changed functions and classes
- Update README.md if necessary
- Add new documentation files if needed

## Review Process

1. A maintainer will review your pull request
2. They may suggest changes or improvements
3. Once approved, your pull request will be merged

## Questions?

If you have questions about contributing, please create an issue with your question.

Thank you for helping improve Communication Evaluator! 