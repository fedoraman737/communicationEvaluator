from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import uuid
import pandas as pd
from werkzeug.utils import secure_filename

# Import local modules
from evaluator.llm_evaluator import LLMEvaluator
from evaluator.batch_evaluator import BatchEvaluator
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
openai_key = os.getenv('OPENAI_API_KEY')
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
api_key_available = bool(openai_key or anthropic_key)

# Use actual API calls if key is available
evaluator = LLMEvaluator(test_mode=not api_key_available)
batch_evaluator = BatchEvaluator(evaluator)

# Log which mode we're using
if not api_key_available:
    app.logger.warning("No API keys found. Running in test mode with sample responses.")
else:
    app.logger.info(f"API key found for provider: {os.getenv('LLM_PROVIDER', 'openai')}. Using actual LLM evaluations.")

# Store batch jobs in memory (in a real app, this would be in a database)
batch_jobs = {}

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

@app.route('/batch')
def batch():
    """Display the batch upload page"""
    return render_template('batch.html', batches=batch_jobs)

@app.route('/upload_batch', methods=['POST'])
def upload_batch():
    """Handle batch file upload and start processing"""
    if 'csv_file' not in request.files:
        flash("No file part", "error")
        return redirect(url_for('batch'))
    
    file = request.files['csv_file']
    
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(url_for('batch'))
    
    if file and (file.filename.endswith(('.csv', '.xlsx', '.xls'))):
        # Create a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        try:
            # Process the CSV file
            app.logger.info(f"Starting batch processing for file: {file_path}")
            batch_id = batch_evaluator.process_csv(file_path)
            
            # Store batch job info
            batch_jobs[batch_id] = {
                'id': batch_id,
                'filename': filename,
                'file_path': file_path,
                'status': 'processing',
                'created_at': datetime.now().isoformat(),
                'result_path': None
            }
            
            flash(f"Batch job started with ID: {batch_id}", "success")
            return redirect(url_for('batch_status', batch_id=batch_id))
            
        except ValueError as e:
            # Handle specific validation errors
            error_message = str(e)
            app.logger.error(f"Validation error processing file: {error_message}")
            flash(f"Error in file format: {error_message}", "error")
            return redirect(url_for('batch'))
        except UnicodeDecodeError as e:
            # Handle encoding errors
            app.logger.error(f"Encoding error processing file: {str(e)}")
            flash("The file contains unsupported characters. Please save your file as UTF-8 encoded CSV.", "error")
            return redirect(url_for('batch'))
        except Exception as e:
            # Handle other errors
            app.logger.error(f"Error processing file: {str(e)}", exc_info=True)
            flash(f"Error processing file: {str(e)}", "error")
            return redirect(url_for('batch'))
    else:
        flash("Invalid file type. Please upload a CSV or Excel file.", "error")
        return redirect(url_for('batch'))

@app.route('/batch_status/<batch_id>')
def batch_status(batch_id):
    """Display the status of a batch job"""
    if batch_id not in batch_jobs:
        flash("Batch job not found", "error")
        return redirect(url_for('batch'))
    
    try:
        # Get batch status from the evaluator
        status = batch_evaluator.get_batch_status(batch_id)
        
        # Update status in our records
        batch_jobs[batch_id]['status'] = status['processing_status']
        
        return render_template('batch_status.html', batch=batch_jobs[batch_id], status=status)
    except Exception as e:
        flash(f"Error getting batch status: {str(e)}", "error")
        return redirect(url_for('batch'))

@app.route('/batch_results/<batch_id>')
def batch_results(batch_id):
    """Display the results of a completed batch job"""
    if batch_id not in batch_jobs:
        flash("Batch job not found", "error")
        return redirect(url_for('batch'))

    batch = batch_jobs[batch_id]

    # Check if we already have results
    if batch['status'] != 'ended':
        flash("Batch processing has not completed yet", "warning")
        return redirect(url_for('batch_status', batch_id=batch_id))

    try:
        # If we haven't processed results yet
        if not batch.get('result_path'):
            # Get the results from the evaluator
            app.logger.info(f"Retrieving batch results for batch ID: {batch_id}")
            evaluations = batch_evaluator.get_batch_results(batch_id, batch['file_path'])

            # Export results to CSV even if empty
            result_filename = f"results_{batch_id}.csv"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)

            if evaluations:
                app.logger.info(f"Exporting {len(evaluations)} evaluations to CSV: {result_path}")
                batch_evaluator.export_results_to_csv(evaluations, result_path)
            else:
                app.logger.warning(f"No successful evaluations to export for batch ID: {batch_id}")
                # Create an empty results file with headers
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write("response_id,empathy_score,positioning_score,persuasion_score,overall_score,strengths,areas_for_improvement,feedback\n")
                    f.write(f"batch_{batch_id},0,0,0,0,\"No successful evaluations\",\"API errors occurred\",\"The batch processing failed. Please try again with fewer responses or check API quotas.\"\n")

            # Update batch info
            batch_jobs[batch_id]['result_path'] = result_path
            batch_jobs[batch_id]['evaluations'] = evaluations

            # If no evaluations were successful, add a warning message
            if not evaluations:
                flash("All evaluation requests in this batch failed. This could be due to API limits or formatting issues.", "warning")
        else:
            # Use cached evaluations if available
            evaluations = batch_jobs[batch_id].get('evaluations', [])

            # If we don't have cached evaluations but have a result path, load from CSV
            if not evaluations and os.path.exists(batch['result_path']):
                # Load evaluations from the CSV
                df = pd.read_csv(batch['result_path'])
                evaluations = []

                for _, row in df.iterrows():
                    # Skip the placeholder row if it exists
                    if str(row.get('response_id', '')).startswith('batch_'):
                        continue

                    try:
                        evaluation = Evaluation(
                            empathy_score=float(row['empathy_score']),
                            positioning_score=float(row['positioning_score']),
                            persuasion_score=float(row['persuasion_score']),
                            overall_score=float(row['overall_score']),
                            strengths=str(row['strengths']).split('; '),
                            areas_for_improvement=str(row['areas_for_improvement']).split('; '),
                            feedback=str(row['feedback']),
                            response_id=str(row['response_id'])
                        )
                        evaluations.append(evaluation)
                    except Exception as e:
                        app.logger.error(f"Error parsing evaluation from CSV: {str(e)}")

                batch_jobs[batch_id]['evaluations'] = evaluations

        return render_template('batch_results.html', batch=batch, evaluations=evaluations)
    except Exception as e:
        app.logger.error(f"Error getting batch results: {str(e)}", exc_info=True)
        flash(f"Error getting batch results: {str(e)}", "error")
        return redirect(url_for('batch'))


@app.route('/print_batch_results/<batch_id>')
@app.route('/print_batch_results/<batch_id>/<advisor_id>')
def print_batch_results(batch_id, advisor_id=None):
    """Display printer-friendly results of a completed batch job, optionally filtered by advisor ID"""
    if batch_id not in batch_jobs:
        flash("Batch job not found", "error")
        return redirect(url_for('batch'))

    batch = batch_jobs[batch_id]

    # Check if we already have results
    if batch['status'] != 'ended':
        flash("Batch processing has not completed yet", "warning")
        return redirect(url_for('batch_status', batch_id=batch_id))

    try:
        # Load the original CSV to get scenario questions
        scenarios = {}
        if os.path.exists(batch['file_path']):
            try:
                # Try to read the original CSV to get scenario text
                if batch['file_path'].endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(batch['file_path'])
                else:
                    df = pd.read_csv(batch['file_path'])

                # Check for special format (scenarios as column headers)
                scenario_columns = []
                for col in df.columns:
                    if isinstance(col, str) and len(col) > 50:
                        scenario_columns.append(col)
                        scenarios[str(len(scenarios) + 1)] = col

                # If we found scenario columns, create a map of scenario_id to text
                if scenario_columns:
                    for i, col in enumerate(scenario_columns, 1):
                        scenarios[str(i)] = col
            except Exception as e:
                app.logger.error(f"Error loading scenarios from CSV: {str(e)}")

        # Use cached evaluations if available or load from result path
        evaluations = batch_jobs[batch_id].get('evaluations', [])

        # If no cached evaluations but have a result path, load from CSV
        if not evaluations and os.path.exists(batch['result_path']):
            # Load evaluations from the CSV
            df = pd.read_csv(batch['result_path'])
            evaluations = []

            for _, row in df.iterrows():
                # Skip placeholder rows
                if str(row.get('response_id', '')).startswith('batch_'):
                    continue

                try:
                    evaluation = Evaluation(
                        empathy_score=float(row['empathy_score']),
                        positioning_score=float(row['positioning_score']),
                        persuasion_score=float(row['persuasion_score']),
                        overall_score=float(row['overall_score']),
                        strengths=str(row['strengths']).split('; '),
                        areas_for_improvement=str(row['areas_for_improvement']).split('; '),
                        feedback=str(row['feedback']),
                        response_id=str(row['response_id'])
                    )
                    evaluations.append(evaluation)
                except Exception as e:
                    app.logger.error(f"Error parsing evaluation from CSV: {str(e)}")

        # Group evaluations by advisor ID if available
        grouped_evaluations = {}
        for eval in evaluations:
            # Extract advisor ID from response_id if possible
            parts = eval.response_id.split('_', 1)
            eval_advisor_id = parts[0] if len(parts) > 1 else 'unknown'

            if eval_advisor_id not in grouped_evaluations:
                grouped_evaluations[eval_advisor_id] = []

            grouped_evaluations[eval_advisor_id].append(eval)

        # If advisor_id is specified, filter to just that advisor's evaluations
        advisor_evals = grouped_evaluations.get(advisor_id, []) if advisor_id else []

        return render_template('print_view.html',
                               batch=batch,
                               evaluations=evaluations,
                               grouped_evaluations=grouped_evaluations,
                               advisor_id=advisor_id,
                               advisor_evals=advisor_evals,
                               scenarios=scenarios)
    except Exception as e:
        app.logger.error(f"Error preparing print view: {str(e)}", exc_info=True)
        flash(f"Error preparing print view: {str(e)}", "error")
        return redirect(url_for('batch_results', batch_id=batch_id))

@app.route('/download_results/<batch_id>')
def download_results(batch_id):
    """Download batch results as CSV"""
    if batch_id not in batch_jobs or not batch_jobs[batch_id].get('result_path'):
        flash("Results not available", "error")
        return redirect(url_for('batch'))
    
    result_path = batch_jobs[batch_id]['result_path']
    
    if not os.path.exists(result_path):
        flash("Result file not found", "error")
        return redirect(url_for('batch_results', batch_id=batch_id))
    
    # In a real app, you'd use send_file to download the file
    # For demo purposes, redirect to batch results
    flash("Download started", "success")
    return redirect(url_for('batch_results', batch_id=batch_id))

@app.route('/api/batch', methods=['POST'])
def api_batch():
    """API endpoint for submitting batch jobs"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and (file.filename.endswith(('.csv', '.xlsx', '.xls'))):
        # Create a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        try:
            # Process the CSV file
            app.logger.info(f"Starting batch processing for file: {file_path}")
            batch_id = batch_evaluator.process_csv(file_path)
            
            # Store batch job info
            batch_jobs[batch_id] = {
                'id': batch_id,
                'filename': filename,
                'file_path': file_path,
                'status': 'processing',
                'created_at': datetime.now().isoformat(),
                'result_path': None
            }
            
            return jsonify({
                'batch_id': batch_id,
                'status': 'processing',
                'status_url': url_for('api_batch_status', batch_id=batch_id, _external=True)
            })
            
        except ValueError as e:
            # Handle specific validation errors
            error_message = str(e)
            app.logger.error(f"Validation error processing file: {error_message}")
            return jsonify({'error': f"Error in file format: {error_message}"}), 400
        except UnicodeDecodeError as e:
            # Handle encoding errors
            app.logger.error(f"Encoding error processing file: {str(e)}")
            return jsonify({'error': "The file contains unsupported characters. Please save your file as UTF-8 encoded CSV."}), 400
        except Exception as e:
            # Handle other errors
            app.logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a CSV or Excel file.'}), 400

@app.route('/api/batch/<batch_id>', methods=['GET'])
def api_batch_status(batch_id):
    """API endpoint for checking batch status"""
    if batch_id not in batch_jobs:
        return jsonify({'error': 'Batch job not found'}), 404
    
    try:
        # Get batch status from the evaluator
        status = batch_evaluator.get_batch_status(batch_id)
        
        # Update status in our records
        batch_jobs[batch_id]['status'] = status['processing_status']
        
        response = {
            'batch_id': batch_id,
            'status': status['processing_status'],
            'created_at': batch_jobs[batch_id]['created_at']
        }
        
        # Add results URL if processing has ended
        if status['processing_status'] == 'ended':
            response['results_url'] = url_for('api_batch_results', batch_id=batch_id, _external=True)
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch/<batch_id>/results', methods=['GET'])
def api_batch_results(batch_id):
    """API endpoint for getting batch results"""
    if batch_id not in batch_jobs:
        return jsonify({'error': 'Batch job not found'}), 404
    
    batch = batch_jobs[batch_id]
    
    # Check if batch processing has completed
    if batch['status'] != 'ended':
        return jsonify({'error': 'Batch processing has not completed yet'}), 400
    
    try:
        # If we haven't processed results yet
        if not batch.get('result_path'):
            # Get the results from the evaluator
            evaluations = batch_evaluator.get_batch_results(batch_id, batch['file_path'])
            
            # Export results to CSV
            result_filename = f"results_{batch_id}.csv"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            
            batch_evaluator.export_results_to_csv(evaluations, result_path)
            
            # Update batch info
            batch_jobs[batch_id]['result_path'] = result_path
            batch_jobs[batch_id]['evaluations'] = evaluations
        else:
            # Use cached evaluations
            evaluations = batch_jobs[batch_id].get('evaluations', [])
        
        # Convert evaluations to JSON
        results = []
        for eval in evaluations:
            results.append({
                'response_id': eval.response_id,
                'empathy_score': eval.empathy_score,
                'positioning_score': eval.positioning_score,
                'persuasion_score': eval.persuasion_score,
                'overall_score': eval.overall_score,
                'strengths': eval.strengths,
                'areas_for_improvement': eval.areas_for_improvement,
                'feedback': eval.feedback
            })
        
        return jsonify({
            'batch_id': batch_id,
            'results': results,
            'download_url': url_for('api_batch_download', batch_id=batch_id, _external=True)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch/<batch_id>/download', methods=['GET'])
def api_batch_download(batch_id):
    """API endpoint for downloading batch results as CSV"""
    if batch_id not in batch_jobs or not batch_jobs[batch_id].get('result_path'):
        return jsonify({'error': 'Results not available'}), 404
    
    # In a real app, you'd return the file for download
    # For demo purposes, just return the URL
    return jsonify({
        'batch_id': batch_id,
        'result_url': url_for('download_results', batch_id=batch_id, _external=True)
    })

if __name__ == '__main__':
    app.run(debug=True) 