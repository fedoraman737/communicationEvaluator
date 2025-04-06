from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
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

# Initialize the LLM evaluator, use test_mode if API keys are not available
api_key_available = bool(os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'))
evaluator = LLMEvaluator(test_mode=not api_key_available)

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
    
    # Evaluate the response using the LLM
    evaluation = evaluator.evaluate_response(response)
    
    # In a real application, save to database
    # For demo, just pass to results page
    return redirect(url_for('evaluation_result', evaluation_id=evaluation.id))

@app.route('/evaluation/<evaluation_id>')
def evaluation_result(evaluation_id):
    # In a real application, fetch from database
    # For demo, create a sample evaluation
    sample_evaluation = Evaluation(
        id=evaluation_id,
        empathy_score=8.5,
        positioning_score=7.2,
        persuasion_score=6.8,
        overall_score=7.5,
        strengths=["Strong acknowledgment of customer concerns", "Clear explanation of policies"],
        areas_for_improvement=["Could improve tone when explaining limitations", "More proactive offering of alternatives"],
        feedback="Your response showed good empathy by acknowledging the customer's frustration. Consider offering alternatives earlier in the conversation to show proactive problem-solving."
    )
    
    return render_template('evaluation_result.html', evaluation=sample_evaluation)

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