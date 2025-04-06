# User Guide: Communication Evaluator

This guide will help you use the Communication Evaluator application effectively.

## Overview

The Communication Evaluator is a tool that analyzes communication responses to provide objective feedback on your communication skills. It evaluates responses based on three key dimensions:

1. **Empathy**: How well you acknowledge and connect with the recipient's emotions
2. **Positioning**: How effectively you balance positive/negative sentiments and maintain appropriate tone
3. **Persuasion**: Your use of persuasive techniques such as social proof and reciprocity

## Getting Started

1. Launch the application by running `python run.py`
2. Open your browser and navigate to http://127.0.0.1:5000
3. You'll see the home page with an introduction to the Communication Evaluator

## Selecting a Scenario

1. Click on "View Scenarios" on the home page
2. Browse the available communication scenarios
3. Click on a scenario to view its details

## Responding to a Scenario

1. On the scenario detail page, you'll see:
   - The scenario title
   - A description of the situation
   - A form to submit your response

2. Enter your advisor ID (optional) and your response to the scenario in the text area
3. Click "Submit Response" to evaluate your communication

## Understanding Your Evaluation Results

After submitting your response, you'll see an evaluation page with:

### Overall Score

Your overall communication effectiveness score out of 10.

### Dimension Scores

Individual scores for each communication dimension:
- **Empathy Score**: How well you connected with emotions
- **Positioning Score**: How well you balanced tone and sentiment
- **Persuasion Score**: How effectively you used persuasion techniques

### Strengths & Areas for Improvement

The evaluation identifies specific strengths in your communication and areas where you can improve.

### Feedback

Personalized feedback with actionable suggestions to enhance your communication skills.

## Example Scenario & Response

**Scenario**: Refund Request Denial

*The customer is nervous after learning their refund request for a $10.99 app is denied. Provide your best response explaining this situation.*

**Sample Response**:

```
I completely understand your concern about the refund for the $10.99 app. I know it's frustrating when you're unable to get a refund for a purchase.

Unfortunately, our refund policy for digital products doesn't allow refunds once the app has been downloaded, as mentioned in our terms of service. This is standard practice for digital goods across the industry.

However, I'd like to help find an alternative solution. I can offer you:
1. A store credit that you can use toward another purchase
2. Technical support if you're having issues with the app
3. Recommendations for similar apps that might better suit your needs

Would any of these options work better for you? I'm committed to finding a solution that leaves you satisfied with our service.
```

**Sample Evaluation**:
- Overall Score: 8.5/10
- Empathy: 9.0
- Positioning: 8.5
- Persuasion: 8.0

## Tips for Higher Scores

### Improving Empathy
- Acknowledge the customer's feelings
- Use phrases that show understanding
- Validate their perspective

### Improving Positioning
- Balance explaining policies with showing care
- Use a supportive tone, even when delivering unwelcome news
- Focus on what you can do, not just what you can't do

### Improving Persuasion
- Offer specific alternatives
- Present options clearly
- Use social proof when appropriate ("many customers find that...")

## Frequently Asked Questions

### How is my response evaluated?
The evaluation uses advanced language models to analyze your response against best practices in professional communication.

### Are my responses stored?
In the current implementation, responses are processed but not permanently stored.

### Can I see previous evaluations?
The current version doesn't support viewing historical evaluations, but this feature may be added in future updates.

### How accurate is the evaluation?
While the system uses sophisticated AI, it should be considered a learning tool rather than an absolute measure of communication quality. 