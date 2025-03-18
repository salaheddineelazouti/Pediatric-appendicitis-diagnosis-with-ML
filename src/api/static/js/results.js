// Results page JavaScript functionality
document.addEventListener('DOMContentLoaded', function() {
    // Set the progress bar widths properly to avoid CSS linting issues
    document.querySelectorAll('[data-progress-value]').forEach(function(bar) {
        const progressValue = bar.getAttribute('data-progress-value');
        // Set a small delay to allow for smooth animation
        setTimeout(function() {
            bar.style.width = (progressValue || 0) + '%';
        }, 100);
    });
    
    // User feedback form handling
    const feedbackForm = document.getElementById('feedbackForm');
    if (feedbackForm) {
        console.log('Feedback form found and initialized');
        
        // Handle the "other diagnosis" field visibility
        const actualDiagnosis = document.getElementById('actualDiagnosis');
        const otherDiagnosis = document.getElementById('otherDiagnosis');
        
        if (actualDiagnosis && otherDiagnosis) {
            actualDiagnosis.addEventListener('change', function() {
                otherDiagnosis.disabled = this.value !== 'autre';
                if (this.value !== 'autre') {
                    otherDiagnosis.value = '';
                }
            });
        }
        
        // Add form validation before submission
        feedbackForm.addEventListener('submit', function(event) {
            let isValid = true;
            
            // Check if diagnostic accuracy is selected
            const diagAccuracy = document.querySelector('input[name="diagnosticAccuracy"]:checked');
            if (!diagAccuracy) {
                isValid = false;
                alert('Veuillez évaluer la précision du diagnostic prédictif.');
            }
            
            // Check if actual diagnosis is selected
            if (actualDiagnosis && actualDiagnosis.value === '') {
                isValid = false;
                alert('Veuillez sélectionner le diagnostic réel.');
            }
            
            // Check if "other diagnosis" is filled when "autre" is selected
            if (actualDiagnosis && actualDiagnosis.value === 'autre' && 
                otherDiagnosis && otherDiagnosis.value.trim() === '') {
                isValid = false;
                alert('Veuillez préciser l\'autre diagnostic.');
            }
            
            // Check if usefulness rating is selected
            const usefulnessRating = document.querySelector('input[name="usefulnessRating"]:checked');
            if (!usefulnessRating) {
                isValid = false;
                alert('Veuillez évaluer l\'utilité des explications fournies.');
            }
            
            // Prevent form submission if validation fails
            if (!isValid) {
                event.preventDefault();
            } else {
                // Show loading spinner or disable button during submission
                const submitButton = feedbackForm.querySelector('button[type="submit"]');
                if (submitButton) {
                    submitButton.disabled = true;
                    submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Envoi en cours...';
                }
                
                // Analytics logging - can be enabled if needed
                console.log('Feedback form submitted successfully');
            }
        });
    }
    
    // Add animation when explanation cards are scrolled into view
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate__animated', 'animate__fadeIn');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.explanation-card').forEach(card => {
        observer.observe(card);
    });
});
