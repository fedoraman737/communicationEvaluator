# API Documentation

The Communication Evaluator provides a RESTful API that allows you to programmatically evaluate communication responses.

## Base URL

When running locally:
```
http://127.0.0.1:5000
```

## Authentication

Currently, the API does not require authentication. In a production environment, you should implement API key authentication.

## Endpoints

### Evaluate a Communication Response

**Endpoint:** `/api/evaluate`

**Method:** POST

**Content-Type:** application/json

**Request Body:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| scenario_id | String | Yes | ID of the scenario being responded to |
| advisor_id | String | No | ID of the person providing the response |
| response_text | String | Yes | The communication response to evaluate |

**Example Request:**

```json
{
  "scenario_id": "1",
  "advisor_id": "user123",
  "response_text": "I understand your frustration with the refund policy. While I can't offer a refund in this case due to our digital product policies, I'd be happy to provide you with a credit for a future purchase or suggest alternative apps that might better meet your needs. Would either of those options work for you?"
}
```

**Example Response:**

```json
{
  "evaluation_id": "550e8400-e29b-41d4-a716-446655440000",
  "empathy_score": 8.5,
  "positioning_score": 7.2,
  "persuasion_score": 6.8,
  "overall_score": 7.5,
  "strengths": [
    "Strong acknowledgment of customer concerns", 
    "Clear explanation of policies"
  ],
  "areas_for_improvement": [
    "Could improve tone when explaining limitations", 
    "More proactive offering of alternatives"
  ],
  "feedback": "Your response showed good empathy by acknowledging the customer's frustration. Consider offering alternatives earlier in the conversation to show proactive problem-solving."
}
```

## Response Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input parameters or missing required fields |
| 500 | Server Error - An error occurred processing the request |

## Rate Limiting

The API currently does not implement rate limiting. However, be aware that the underlying LLM APIs (OpenAI, Anthropic) have their own rate limits and usage quotas.

## Error Handling

**Example Error Response:**

```json
{
  "error": "Missing required fields"
}
```

## Implementation Example

### Python

```python
import requests
import json

url = "http://127.0.0.1:5000/api/evaluate"
payload = {
  "scenario_id": "1",
  "advisor_id": "user123",
  "response_text": "Your communication response here"
}
headers = {
  "Content-Type": "application/json"
}

response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json())
```

### JavaScript

```javascript
fetch('http://127.0.0.1:5000/api/evaluate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    scenario_id: "1",
    advisor_id: "user123",
    response_text: "Your communication response here"
  }),
})
.then(response => response.json())
.then(data => console.log(data))
.catch((error) => console.error('Error:', error));
``` 