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

# Initialize the LLM evaluator
# test_mode = os.getenv('TEST_MODE', 'False').lower() == 'true' # No longer needed
# evaluator = LLMEvaluator(test_mode=test_mode) # No longer needed
# batch_evaluator = BatchEvaluator(evaluator) # No longer needed

# Store evaluations temporarily in memory (replace with DB in production)
# evaluation_cache: Dict[str, Evaluation] = {} # No longer needed

# Log which mode we're using
# if test_mode: # No longer needed
#     app.logger.warning("Running in test mode with sample responses.") # No longer needed
# else: # No longer needed
#     app.logger.info("Using DeepSeek for evaluations.") # No longer needed

# Store batch jobs in memory (in a real app, this would be in a database)
# batch_jobs = {} # No longer needed

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

# All routes related to submission, evaluation, and batch processing will be removed.

if __name__ == '__main__':
    app.run(debug=True) 