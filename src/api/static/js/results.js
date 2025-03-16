// Results page JavaScript functionality
document.addEventListener('DOMContentLoaded', function() {
    // You can add interactive elements .here if needed
    console.log('Results page loaded');
    
    // Set the progress bar widths properly to avoid CSS linting issues
    document.querySelectorAll('[data-progress-value]').forEach(function(bar) {
        const progressValue = bar.getAttribute('data-progress-value');
        // Set a small delay to allow for smooth animation
        setTimeout(function() {
            bar.style.width = (progressValue || 0) + '%';
        }, 100);
    });
});
