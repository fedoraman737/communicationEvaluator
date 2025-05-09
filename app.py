from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import uuid
import pandas as pd
from werkzeug.utils import secure_filename
from typing import Dict

# Import local modules
from models.scenario import Scenario
from models.response import Response
from models.evaluation import Evaluation
from evaluator.rule_based_evaluator import RuleBasedEvaluator

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev_secret_key')

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Initialize the RuleBasedEvaluator
evaluator = RuleBasedEvaluator()

# Store evaluations temporarily in memory
evaluation_cache: Dict[str, Evaluation] = {}

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
            description="Craft your best response to a customer asking: 'Why do we have to wait 20 days for account recovery to reset my password?'",
            category="Account Management"
        ),
        Scenario(
            id=3,
            title="Troubleshooting Skepticism",
            description="During troubleshooting for a slow computer, the customer says: 'I don't think this troubleshooting will work.' What is your best response?",
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
            description="Craft your best response to a customer asking: 'Why do we have to wait 20 days for account recovery to reset my password?'",
            category="Account Management"
        ),
        3: Scenario(
            id=3,
            title="Troubleshooting Skepticism",
            description="During troubleshooting for a slow computer, the customer says: 'I don't think this troubleshooting will work.' What is your best response?",
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
    advisor_id = request.form.get('advisor_id', 'default_advisor')
    response_text = request.form.get('response_text')
    
    if not all([scenario_id, response_text]):
        flash("Scenario ID and Response Text are required", "error")
        if scenario_id:
            return redirect(url_for('scenario_detail', scenario_id=int(scenario_id)))
        else:
            return redirect(url_for('scenarios'))
    
    response_internal_id = f"{advisor_id}_{scenario_id}_{uuid.uuid4()}"
    response = Response(
        id=response_internal_id,
        advisor_id=advisor_id,
        scenario_id=scenario_id,
        text=response_text,
        submitted_at=datetime.now()
    )
    
    evaluation = evaluator.evaluate_response(response)
    
    evaluation_cache[evaluation.id] = evaluation
    app.logger.info(f"Stored evaluation {evaluation.id} in cache.")
    
    return redirect(url_for('evaluation_result', evaluation_id=evaluation.id))

@app.route('/evaluation/<evaluation_id>')
def evaluation_result(evaluation_id):
    evaluation = evaluation_cache.get(evaluation_id)
    
    if not evaluation:
        app.logger.error(f"Evaluation ID {evaluation_id} not found in cache.")
        flash("Evaluation results not found or have expired.", "error")
        return redirect(url_for('scenarios'))

    app.logger.info(f"Retrieved evaluation {evaluation_id} from cache.")
    return render_template('evaluation_result.html', evaluation=evaluation)

if __name__ == '__main__':
    app.run(debug=True) 