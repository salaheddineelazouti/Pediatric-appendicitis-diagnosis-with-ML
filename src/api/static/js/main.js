/**
 * Pediatric Appendicitis Diagnosis Support Tool
 * Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltips.length > 0) {
        tooltips.forEach(tooltip => {
            new bootstrap.Tooltip(tooltip);
        });
    }

    // Current year for footer copyright
    const currentYearElement = document.querySelector('.current-year');
    if (currentYearElement) {
        currentYearElement.textContent = new Date().getFullYear();
    }

    // Handle diagnosis form enhanced functionality
    const diagnosisForm = document.getElementById('diagnosisForm');
    if (diagnosisForm) {
        setupDiagnosisForm();
    }

    // Handle results page functionality
    const resultsPage = document.querySelector('.results-container');
    if (resultsPage) {
        setupResultsPage();
    }
});

/**
 * Set up enhanced functionality for the diagnosis form
 */
function setupDiagnosisForm() {
    // Calculate BMI when weight or height changes
    const weightInput = document.getElementById('weight');
    const heightInput = document.getElementById('height');
    const bmiInput = document.getElementById('bmi');
    
    function calculateBMI() {
        if (weightInput.value && heightInput.value) {
            const weight = parseFloat(weightInput.value);
            const height = parseFloat(heightInput.value) / 100; // Convert cm to m
            if (weight > 0 && height > 0) {
                const bmi = (weight / (height * height)).toFixed(1);
                bmiInput.value = bmi;
            }
        } else {
            bmiInput.value = '';
        }
    }
    
    if (weightInput && heightInput && bmiInput) {
        weightInput.addEventListener('input', calculateBMI);
        heightInput.addEventListener('input', calculateBMI);
    }
    
    // Toggle ultrasound findings based on ultrasound performed
    const ultrasoundPerformedSelect = document.getElementById('ultrasound_performed');
    const ultrasoundFindingSelect = document.getElementById('ultrasound_finding');
    const ultrasoundDetailsInputs = document.querySelectorAll('.ultrasound-details input');
    
    if (ultrasoundPerformedSelect && ultrasoundFindingSelect) {
        ultrasoundPerformedSelect.addEventListener('change', function() {
            const isPerformed = this.value === 'yes';
            ultrasoundFindingSelect.disabled = !isPerformed;
            
            if (!isPerformed) {
                ultrasoundFindingSelect.value = '';
                ultrasoundDetailsInputs.forEach(input => {
                    input.disabled = true;
                    input.checked = false;
                });
            } else {
                ultrasoundDetailsInputs.forEach(input => {
                    input.disabled = false;
                });
            }
        });
    }
    
    // Toggle CT findings based on CT performed
    const ctPerformedSelect = document.getElementById('ct_performed');
    const ctFindingSelect = document.getElementById('ct_finding');
    
    if (ctPerformedSelect && ctFindingSelect) {
        ctPerformedSelect.addEventListener('change', function() {
            const isPerformed = this.value === 'yes';
            ctFindingSelect.disabled = !isPerformed;
            
            if (!isPerformed) {
                ctFindingSelect.value = '';
            }
        });
    }
    
    // Form validation
    const form = document.getElementById('diagnosisForm');
    if (form) {
        form.addEventListener('submit', function(event) {
            const age = document.getElementById('age').value;
            const gender = document.getElementById('gender').value;
            const painDuration = document.getElementById('duration_of_pain').value;
            
            if (!age || !gender || !painDuration) {
                event.preventDefault();
                alert('Please fill in all required fields');
            }
        });
    }

    // Add visual feedback for form fields
    const formInputs = document.querySelectorAll('.form-control, .form-select');
    formInputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.classList.add('border-primary');
        });
        
        input.addEventListener('blur', function() {
            this.classList.remove('border-primary');
        });
    });
}

/**
 * Set up functionality for the results page
 */
function setupResultsPage() {
    // Print functionality
    const printButton = document.querySelector('.btn-print-report');
    if (printButton) {
        printButton.addEventListener('click', function() {
            window.print();
        });
    }

    // New assessment button
    const newAssessmentButton = document.querySelector('.btn-new-assessment');
    if (newAssessmentButton) {
        newAssessmentButton.addEventListener('click', function() {
            window.location.href = this.getAttribute('data-href');
        });
    }
}

/**
 * Create a chart for displaying feature importance
 * @param {string} chartId - The ID of the canvas element
 * @param {Array} features - Array of feature names
 * @param {Array} values - Array of feature importance values
 */
function createFeatureImportanceChart(chartId, features, values) {
    const ctx = document.getElementById(chartId);
    if (!ctx) return;
    
    // Determine colors based on positive/negative values
    const backgroundColors = values.map(value => 
        value >= 0 ? 'rgba(40, 167, 69, 0.7)' : 'rgba(220, 53, 69, 0.7)'
    );
    
    const borderColors = values.map(value => 
        value >= 0 ? 'rgb(40, 167, 69)' : 'rgb(220, 53, 69)'
    );
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Feature Importance',
                data: values,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += context.raw.toFixed(3);
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}
