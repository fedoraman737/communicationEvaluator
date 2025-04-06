// Main JavaScript for Communication Evaluator

document.addEventListener('DOMContentLoaded', function() {
    // Auto-dismiss alerts after 5 seconds
    setTimeout(function() {
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);

    // Character counter for response textarea
    const responseTextarea = document.getElementById('response_text');
    if (responseTextarea) {
        const createCounter = function() {
            const counterDiv = document.createElement('div');
            counterDiv.className = 'text-muted small mt-1';
            counterDiv.id = 'char-counter';
            responseTextarea.parentNode.insertBefore(counterDiv, responseTextarea.nextSibling);
            return counterDiv;
        };

        const counterDiv = createCounter();
        
        const updateCounter = function() {
            const charCount = responseTextarea.value.length;
            counterDiv.textContent = `${charCount} characters`;
            
            // Give visual feedback for very short responses
            if (charCount < 50) {
                counterDiv.className = 'text-danger small mt-1';
                counterDiv.textContent += ' (Your response may be too short for effective evaluation)';
            } else {
                counterDiv.className = 'text-muted small mt-1';
            }
        };
        
        responseTextarea.addEventListener('input', updateCounter);
        // Initial count
        updateCounter();
    }
}); 