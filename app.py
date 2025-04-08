from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Import local modules
from evaluator.llm_evaluator import LLMEvaluator
from models.scenario import Scenario
from models.response import Response
from models.evaluation import Evaluation

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev_secret_key')

# Initialize the LLM evaluator
openai_key = os.getenv('OPENAI_API_KEY')
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
api_key_available = bool(openai_key or anthropic_key)

# Only use test mode if no API key is available
evaluator = LLMEvaluator(test_mode=not api_key_available)

# Log which mode we're using
if not api_key_available:
    app.logger.warning("No API keys found. Running in test mode with sample responses.")
else:
    app.logger.info(f"API key found for provider: {os.getenv('LLM_PROVIDER', 'openai')}. Using actual LLM evaluations.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scenarios')
def scenarios():
    # In a real application, these would be loaded from a database
    # For demo purposes, we'll use hardcoded scenarios
    demo_scenarios = [
        Scenario(
            id=1,
            title="Refund Request Denial",
            description="The customer is nervous after learning their refund request for a $10.99 app is denied. Provide your best response explaining this situation.",
            category="Customer Support"
        ),
        Scenario(
            id=2,
            title="Account Recovery Wait Time",
            description="Craft your best response to a customer asking: 'Why do we have to wait 20 days for Apple Account recovery to reset my password?'",
            category="Account Management"
        ),
        Scenario(
            id=3,
            title="Troubleshooting Skepticism",
            description="During troubleshooting for a slow MacBook, the customer says: 'I don't think this troubleshooting will work.' What is your best response?",
            category="Technical Support"
        )
    ]
    return render_template('scenarios.html', scenarios=demo_scenarios)

@app.route('/scenario/<int:scenario_id>')
def scenario_detail(scenario_id):
    # In a real application, fetch from database
    demo_scenarios = {
        1: Scenario(
            id=1,
            title="Refund Request Denial",
            description="The customer is nervous after learning their refund request for a $10.99 app is denied. Provide your best response explaining this situation.",
            category="Customer Support"
        ),
        2: Scenario(
            id=2,
            title="Account Recovery Wait Time",
            description="Craft your best response to a customer asking: 'Why do we have to wait 20 days for Apple Account recovery to reset my password?'",
            category="Account Management"
        ),
        3: Scenario(
            id=3,
            title="Troubleshooting Skepticism",
            description="During troubleshooting for a slow MacBook, the customer says: 'I don't think this troubleshooting will work.' What is your best response?",
            category="Technical Support"
        )
    }
    scenario = demo_scenarios.get(scenario_id)
    if not scenario:
        flash("Scenario not found", "error")
        return redirect(url_for('scenarios'))
    
    return render_template('scenario_detail.html', scenario=scenario)

@app.route('/submit_response', methods=['POST'])
def submit_response():
    scenario_id = request.form.get('scenario_id')
    advisor_id = request.form.get('advisor_id')
    response_text = request.form.get('response_text')
    
    if not all([scenario_id, advisor_id, response_text]):
        flash("All fields are required", "error")
        return redirect(url_for('scenario_detail', scenario_id=scenario_id))
    
    # Create a new response object
    response = Response(
        advisor_id=advisor_id,
        scenario_id=scenario_id,
        text=response_text,
        submitted_at=datetime.now()
    )
    
    # Store this response in the session for use in evaluation_result
    session['last_response_text'] = response_text
    session['last_scenario_id'] = scenario_id
    session['last_advisor_id'] = advisor_id
    
    # Evaluate the response using the LLM
    evaluation = evaluator.evaluate_response(response)
    
    # In a real application, save to database
    # For demo, just pass to results page
    return redirect(url_for('evaluation_result', evaluation_id=evaluation.id))

@app.route('/evaluation/<evaluation_id>')
def evaluation_result(evaluation_id):
    # In a real application, fetch from database
    # For demo purposes, try to get the response details from the session
    # If not available, use a placeholder

    # Check if we have stored response details
    response_text = session.get('last_response_text', "This is a placeholder for the actual response that was evaluated.")
    scenario_id = int(session.get('last_scenario_id', 1))
    advisor_id = session.get('last_advisor_id', "anonymous")
    
    # Create a response object with the actual text if available
    response = Response(
        advisor_id=advisor_id,
        scenario_id=scenario_id,
        text=response_text,
        submitted_at=datetime.now()
    )
    
    # Get actual evaluation from the LLM
    evaluation = evaluator.evaluate_response(response)
    
    # Set the passed ID to maintain the route parameter
    evaluation.id = evaluation_id
    
    return render_template('evaluation_result.html', evaluation=evaluation)

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    data = request.json
    
    if not data or 'response_text' not in data or 'scenario_id' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    response = Response(
        advisor_id=data.get('advisor_id', 'anonymous'),
        scenario_id=data['scenario_id'],
        text=data['response_text'],
        submitted_at=datetime.now()
    )
    
    evaluation = evaluator.evaluate_response(response)
    
    return jsonify({
        'evaluation_id': evaluation.id,
        'empathy_score': evaluation.empathy_score,
        'positioning_score': evaluation.positioning_score,
        'persuasion_score': evaluation.persuasion_score,
        'overall_score': evaluation.overall_score,
        'strengths': evaluation.strengths,
        'areas_for_improvement': evaluation.areas_for_improvement,
        'feedback': evaluation.feedback
    })

if __name__ == '__main__':
    app.run(debug=True) 