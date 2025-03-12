# Changelog

All notable changes to the Pediatric Appendicitis Diagnosis project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
