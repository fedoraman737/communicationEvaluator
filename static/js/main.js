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

    // animation for main content elements
    const contentElements = document.querySelectorAll('.card, .display-5, .lead, .btn-lg');
    contentElements.forEach((element, index) => {
        setTimeout(() => {
            element.classList.add('fade-in');
        }, index * 100);
    });

    // counter for response textarea
    const responseTextarea = document.getElementById('response_text');
    if (responseTextarea) {
        const createCounter = function() {
            const counterWrapper = document.createElement('div');
            counterWrapper.className = 'mt-2 d-flex align-items-center justify-content-between';
            
            const counterDiv = document.createElement('div');
            counterDiv.className = 'text-muted small';
            counterDiv.id = 'char-counter';
            
            const progressContainer = document.createElement('div');
            progressContainer.className = 'progress flex-grow-1 mx-2';
            progressContainer.style.height = '6px';
            
            const progressBar = document.createElement('div');
            progressBar.className = 'progress-bar';
            progressBar.setAttribute('role', 'progressbar');
            progressBar.setAttribute('aria-valuemin', '0');
            progressBar.setAttribute('aria-valuemax', '100');
            
            progressContainer.appendChild(progressBar);
            
            counterWrapper.appendChild(counterDiv);
            counterWrapper.appendChild(progressContainer);
            
            responseTextarea.parentNode.insertBefore(counterWrapper, responseTextarea.nextSibling);
            
            return {
                counter: counterDiv,
                progress: progressBar
            };
        };

        const counterElements = createCounter();
        
        const updateCounter = function() {
            const charCount = responseTextarea.value.length;
            const optimal = 150;
            const percentage = Math.min(100, Math.round(charCount / optimal * 100));
            
            counterElements.counter.textContent = `${charCount} characters`;
            counterElements.progress.style.width = `${percentage}%`;
            counterElements.progress.setAttribute('aria-valuenow', percentage);
            
            // provides feedback based on response length
            if (charCount < 50) {
                counterElements.counter.className = 'text-danger small';
                counterElements.counter.textContent += ' (Too short for effective evaluation)';
                counterElements.progress.className = 'progress-bar bg-danger';
            } else if (charCount < 100) {
                counterElements.counter.className = 'text-warning small';
                counterElements.counter.textContent += ' (Consider adding more detail)';
                counterElements.progress.className = 'progress-bar bg-warning';
            } else if (charCount > 500) {
                counterElements.counter.className = 'text-info small';
                counterElements.counter.textContent += ' (Very detailed response)';
                counterElements.progress.className = 'progress-bar bg-info';
            } else {
                counterElements.counter.className = 'text-success small';
                counterElements.progress.className = 'progress-bar bg-success';
            }
        };
        
        responseTextarea.addEventListener('input', updateCounter);
        // initial count
        updateCounter();
    }
}); 