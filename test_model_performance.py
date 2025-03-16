"""
Test Model Performance for Pediatric Appendicitis Diagnosis
----------------------------------------------------------
This script tests the model with diverse clinical scenarios to verify its performance,
including edge cases, typical cases, and challenging diagnostic situations.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from tabulate import tabulate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join('models', 'best_model_retrained.pkl')  # Current app model
FIXED_MODEL_PATH = os.path.join('models', 'best_model_fixed.pkl')  # New fixed model

def load_models():
    """Load both the current and fixed models for comparison"""
    models = {}
    
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                models['current'] = pickle.load(f)
            logger.info(f"Current model loaded: {type(models['current']).__name__}")
        else:
            logger.warning(f"Current model not found at {MODEL_PATH}")

        if os.path.exists(FIXED_MODEL_PATH):
            with open(FIXED_MODEL_PATH, 'rb') as f:
                models['fixed'] = pickle.load(f)
            logger.info(f"Fixed model loaded: {type(models['fixed']).__name__}")
        else:
            logger.warning(f"Fixed model not found at {FIXED_MODEL_PATH}")
            
        return models
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return {}

def generate_test_cases():
    """Generate diverse test cases for model evaluation"""
    test_cases = []
    
    # Classic appendicitis case - high scores, all symptoms present
    test_cases.append({
        "name": "Classic appendicitis",
        "features": {
            "age": 10,
            "gender": 1,  # Male
            "duration": 36,  # 36 hours of pain
            "migration": 1,  # Pain migration present
            "anorexia": 1,  # Loss of appetite
            "nausea": 1,
            "vomiting": 1,
            "right_lower_quadrant_pain": 1,  # RLQ pain
            "fever": 1,  # Fever present
            "rebound_tenderness": 1,  # Rebound tenderness present
            "white_blood_cell_count": 18.5,  # Elevated WBC
            "neutrophil_percentage": 88.0,  # Elevated neutrophils
            "c_reactive_protein": 120.0,  # Elevated CRP
            "pediatric_appendicitis_score": 9,  # High PAS
            "alvarado_score": 9  # High Alvarado
        },
        "expected": 1  # Appendicitis
    })
    
    # Definite non-appendicitis - gastroenteritis
    test_cases.append({
        "name": "Gastroenteritis",
        "features": {
            "age": 8,
            "gender": 0,  # Female
            "duration": 12,
            "migration": 0,  # No migration
            "anorexia": 1,
            "nausea": 1,
            "vomiting": 1,  # Vomiting present
            "right_lower_quadrant_pain": 0,  # Diffuse pain, not RLQ
            "fever": 1,  # May have fever
            "rebound_tenderness": 0,  # No rebound
            "white_blood_cell_count": 9.5,  # Normal/slightly elevated
            "neutrophil_percentage": 65.0,  # Normal/slightly elevated
            "c_reactive_protein": 15.0,  # Mild elevation
            "pediatric_appendicitis_score": 3,
            "alvarado_score": 4
        },
        "expected": 0  # Not appendicitis
    })
    
    # Early appendicitis - some symptoms but not all
    test_cases.append({
        "name": "Early appendicitis",
        "features": {
            "age": 12,
            "gender": 1,  # Male
            "duration": 12,  # Early in course
            "migration": 1,  # Has migration
            "anorexia": 1,
            "nausea": 1,
            "vomiting": 0,  # No vomiting yet
            "right_lower_quadrant_pain": 1,
            "fever": 0,  # No fever yet
            "rebound_tenderness": 0,  # No rebound yet
            "white_blood_cell_count": 12.0,  # Slightly elevated WBC
            "neutrophil_percentage": 75.0,  # Moderately elevated neutrophils
            "c_reactive_protein": 30.0,  # Moderately elevated CRP
            "pediatric_appendicitis_score": 5,  # Moderate PAS
            "alvarado_score": 6  # Moderate Alvarado
        },
        "expected": 1  # Early appendicitis, should still detect
    })
    
    # Constipation mimicking appendicitis
    test_cases.append({
        "name": "Constipation",
        "features": {
            "age": 9,
            "gender": 0,  # Female
            "duration": 48,  # Longer duration
            "migration": 0,  # No migration
            "anorexia": 0,
            "nausea": 0,
            "vomiting": 0,
            "right_lower_quadrant_pain": 1,  # RLQ pain can be present
            "fever": 0,  # No fever
            "rebound_tenderness": 0,  # No rebound
            "white_blood_cell_count": 8.0,  # Normal WBC
            "neutrophil_percentage": 50.0,  # Normal neutrophils
            "c_reactive_protein": 5.0,  # Normal CRP
            "pediatric_appendicitis_score": 2,
            "alvarado_score": 2
        },
        "expected": 0  # Not appendicitis
    })
    
    # Complicated appendicitis - perforated
    test_cases.append({
        "name": "Perforated appendicitis",
        "features": {
            "age": 7,
            "gender": 1,  # Male
            "duration": 72,  # Long duration
            "migration": 1,
            "anorexia": 1,
            "nausea": 1,
            "vomiting": 1,
            "right_lower_quadrant_pain": 1,
            "fever": 1,  # High fever
            "rebound_tenderness": 1,  # Significant rebound
            "white_blood_cell_count": 22.0,  # Very high WBC
            "neutrophil_percentage": 92.0,  # Very high neutrophils
            "c_reactive_protein": 200.0,  # Very high CRP
            "pediatric_appendicitis_score": 10,
            "alvarado_score": 10
        },
        "expected": 1  # Definite appendicitis
    })
    
    # Ovarian pathology (girls) - might be confused with appendicitis
    test_cases.append({
        "name": "Ovarian pathology",
        "features": {
            "age": 14,
            "gender": 0,  # Female
            "duration": 24,
            "migration": 0,
            "anorexia": 0,
            "nausea": 1,
            "vomiting": 0,
            "right_lower_quadrant_pain": 1,  # RLQ pain
            "fever": 0,
            "rebound_tenderness": 1,  # May have rebound
            "white_blood_cell_count": 11.0,  # Mildly elevated
            "neutrophil_percentage": 70.0,  # Mildly elevated
            "c_reactive_protein": 15.0,  # Mildly elevated
            "pediatric_appendicitis_score": 4,
            "alvarado_score": 5
        },
        "expected": 0  # Not appendicitis (though challenging)
    })
    
    # Mesenteric adenitis - common mimic
    test_cases.append({
        "name": "Mesenteric adenitis",
        "features": {
            "age": 6,
            "gender": 1,  # Male
            "duration": 36,
            "migration": 0,
            "anorexia": 1,
            "nausea": 0,
            "vomiting": 0,
            "right_lower_quadrant_pain": 1,  # RLQ pain
            "fever": 1,  # Fever often present
            "rebound_tenderness": 0,
            "white_blood_cell_count": 13.0,  # Moderately elevated
            "neutrophil_percentage": 65.0,  # Mildly elevated
            "c_reactive_protein": 30.0,  # Moderately elevated
            "pediatric_appendicitis_score": 4,
            "alvarado_score": 4
        },
        "expected": 0  # Not appendicitis
    })
    
    # Urinary tract infection
    test_cases.append({
        "name": "Urinary tract infection",
        "features": {
            "age": 5,
            "gender": 0,  # Female
            "duration": 24,
            "migration": 0,
            "anorexia": 0,
            "nausea": 0,
            "vomiting": 0,
            "right_lower_quadrant_pain": 0,  # More suprapubic
            "fever": 1,
            "rebound_tenderness": 0,
            "white_blood_cell_count": 14.0,  # Elevated
            "neutrophil_percentage": 75.0,  # Elevated
            "c_reactive_protein": 50.0,  # Elevated
            "pediatric_appendicitis_score": 3,
            "alvarado_score": 3
        },
        "expected": 0  # Not appendicitis
    })
    
    # Borderline case - could go either way
    test_cases.append({
        "name": "Borderline case",
        "features": {
            "age": 10,
            "gender": 1,  # Male
            "duration": 24,
            "migration": 1,
            "anorexia": 1,
            "nausea": 0,
            "vomiting": 0,
            "right_lower_quadrant_pain": 1,
            "fever": 0,
            "rebound_tenderness": 1,
            "white_blood_cell_count": 11.5,  # Mildly elevated
            "neutrophil_percentage": 72.0,  # Mildly elevated
            "c_reactive_protein": 25.0,  # Mildly elevated
            "pediatric_appendicitis_score": 6,  # Moderate
            "alvarado_score": 6  # Moderate
        },
        "expected": 1  # Likely appendicitis, though borderline
    })
    
    # Add more test cases as needed...
    
    return test_cases

def test_model_with_cases(models, test_cases):
    """Test models with the generated test cases"""
    if not models:
        logger.error("No models available for testing")
        return
    
    results = []
    
    for case in test_cases:
        case_result = {
            "Name": case["name"],
            "Expected": "Appendicitis" if case["expected"] == 1 else "Not Appendicitis"
        }
        
        # Create DataFrame from features
        features_df = pd.DataFrame([case["features"]])
        
        # Test with each available model
        for model_name, model in models.items():
            try:
                prediction = model.predict(features_df)[0]
                probability = model.predict_proba(features_df)[0, 1]
                
                # Add results to dictionary
                case_result[f"{model_name.capitalize()} Prediction"] = "Appendicitis" if prediction == 1 else "Not Appendicitis"
                case_result[f"{model_name.capitalize()} Probability"] = f"{probability:.2f}"
                case_result[f"{model_name.capitalize()} Correct"] = prediction == case["expected"]
            except Exception as e:
                logger.error(f"Error testing {model_name} model on {case['name']}: {str(e)}")
                case_result[f"{model_name.capitalize()} Prediction"] = "Error"
                case_result[f"{model_name.capitalize()} Probability"] = "Error"
                case_result[f"{model_name.capitalize()} Correct"] = False
        
        results.append(case_result)
    
    return results

def analyze_results(results):
    """Analyze and summarize test results"""
    if not results:
        logger.error("No results to analyze")
        return
    
    # Print results as a table
    logger.info("\n" + "="*100)
    logger.info("Test Results Summary")
    logger.info("="*100)
    
    # Get all columns for headers
    headers = list(results[0].keys())
    
    # Create table with proper format for tabulate
    table_data = []
    for res in results:
        row = [res[h] for h in headers]
        table_data.append(row)
    
    # Print table
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    logger.info("\n" + table)
    
    # Calculate accuracy for each model
    model_stats = {}
    model_types = set()
    
    for res in results:
        for key in res.keys():
            if key.endswith(" Correct"):
                model_name = key.replace(" Correct", "").lower()
                model_types.add(model_name)
    
    # Initialize stats
    for model_name in model_types:
        model_stats[model_name] = {
            "total": 0,
            "correct": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 0,
            "true_negatives": 0
        }
    
    # Calculate statistics
    for res in results:
        for model_name in model_types:
            correct_key = f"{model_name.capitalize()} Correct"
            pred_key = f"{model_name.capitalize()} Prediction"
            
            if correct_key in res and pred_key in res:
                model_stats[model_name]["total"] += 1
                
                if res[correct_key]:
                    model_stats[model_name]["correct"] += 1
                
                # Calculate confusion matrix stats
                if res["Expected"] == "Appendicitis" and res[pred_key] == "Appendicitis":
                    model_stats[model_name]["true_positives"] += 1
                elif res["Expected"] == "Not Appendicitis" and res[pred_key] == "Not Appendicitis":
                    model_stats[model_name]["true_negatives"] += 1
                elif res["Expected"] == "Not Appendicitis" and res[pred_key] == "Appendicitis":
                    model_stats[model_name]["false_positives"] += 1
                elif res["Expected"] == "Appendicitis" and res[pred_key] == "Not Appendicitis":
                    model_stats[model_name]["false_negatives"] += 1
    
    # Print model statistics
    logger.info("\n" + "="*50)
    logger.info("Model Performance Comparison")
    logger.info("="*50)
    
    for model_name, stats in model_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        sensitivity = stats["true_positives"] / (stats["true_positives"] + stats["false_negatives"]) if (stats["true_positives"] + stats["false_negatives"]) > 0 else 0
        specificity = stats["true_negatives"] / (stats["true_negatives"] + stats["false_positives"]) if (stats["true_negatives"] + stats["false_positives"]) > 0 else 0
        
        logger.info(f"\n{model_name.capitalize()} Model:")
        logger.info(f"  Accuracy: {accuracy:.2f} ({stats['correct']}/{stats['total']})")
        logger.info(f"  Sensitivity: {sensitivity:.2f}")
        logger.info(f"  Specificity: {specificity:.2f}")
        logger.info(f"  True Positives: {stats['true_positives']}")
        logger.info(f"  True Negatives: {stats['true_negatives']}")
        logger.info(f"  False Positives: {stats['false_positives']}")
        logger.info(f"  False Negatives: {stats['false_negatives']}")
    
    # Identify challenging cases
    logger.info("\n" + "="*50)
    logger.info("Challenging Cases")
    logger.info("="*50)
    
    for res in results:
        is_challenging = False
        for model_name in model_types:
            correct_key = f"{model_name.capitalize()} Correct"
            if correct_key in res and not res[correct_key]:
                is_challenging = True
                break
        
        if is_challenging:
            logger.info(f"\nCase: {res['Name']} (Expected: {res['Expected']})")
            for model_name in model_types:
                pred_key = f"{model_name.capitalize()} Prediction"
                prob_key = f"{model_name.capitalize()} Probability"
                if pred_key in res and prob_key in res:
                    logger.info(f"  {model_name.capitalize()}: {res[pred_key]} (probability: {res[prob_key]})")

def main():
    """Main function to test model performance"""
    logger.info("Starting model testing process...")
    
    # Load models
    models = load_models()
    
    if not models:
        logger.error("No models could be loaded. Exiting.")
        return
    
    # Generate test cases
    test_cases = generate_test_cases()
    logger.info(f"Generated {len(test_cases)} test cases for evaluation")
    
    # Test models with cases
    results = test_model_with_cases(models, test_cases)
    
    # Analyze results
    analyze_results(results)
    
    logger.info("Model testing completed")

if __name__ == "__main__":
    main()
