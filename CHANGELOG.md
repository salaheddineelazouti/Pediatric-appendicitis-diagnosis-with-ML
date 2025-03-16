# Changelog

All notable changes to the Pediatric Appendicitis Diagnosis project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New error handling in diagnose route to return form with error message instead of error page
- Enhanced unit tests for SHAP explainer initialization
- Added `explain` method to `ShapExplainer` class to provide structured explanation data
- Added unit test for the new `explain` method

### Fixed
- Fixed error handling in the diagnose route to improve user experience
- Corrected test_advanced_outlier_detection to properly unpack the tuple returned by the function
- Fixed test_explainer_initialization to properly mock the ShapExplainer class
- Resolved all failing CI tests related to SHAP explainer integration
- Fixed SHAP explainer integration in the `/diagnose` endpoint
- Updated template to display "Prediction Explanation" text in the UI
- Modified `initialize_explainer` method to support force creating new explainer instances for testing

### Changed
- Improved error messages and user feedback on form submission errors

## [1.0.0] - 2025-03-10

### Added
- Initial release of Pediatric Appendicitis Diagnosis tool
- Machine learning model for predicting appendicitis risk
- Web UI for inputting patient data and viewing diagnosis results
- SHAP-based explainability for predictions
- Integration with AI assistant for medical explanations

## [0.1.0] - 2025-03-12

### Added
- Initial release of the Pediatric Appendicitis Diagnosis application
- SVM model implementation with explainability features
- Flask web interface for clinical use
- Medical report generation functionality
- Docker and Docker Compose configuration
- Model verification and integration checks
- Synthetic data generation for model training

### Changed
- Converted prediction model from RandomForest to SVM based on performance testing
- Enhanced feature importance analysis using correlation with target variable
- Improved medical reporting with detailed clinical recommendations

### Fixed
- Addressed missing training data issue by generating synthetic dataset
- Corrected model type to use SVM as recommended by comparative tests
