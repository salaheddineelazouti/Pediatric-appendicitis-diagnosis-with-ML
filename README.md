# Pediatric Appendicitis Diagnosis with Explainable ML

## Project Overview
This project develops a clinical decision-support application aimed at assisting pediatricians in accurately diagnosing appendicitis in children. By leveraging machine learning techniques (specifically Support Vector Machines) and providing transparent, explainable predictions through SHAP (SHapley Additive exPlanations), this tool helps medical professionals make informed decisions based on symptoms and clinical test results.

## Goals
- Develop a robust, explainable machine learning model for appendicitis diagnosis
- Ensure transparency of model predictions using SHAP explainability
- Create an intuitive user interface for medical professionals (Streamlit)
- Follow professional software development practices including CI/CD
- Document AI-assisted development through prompt engineering

## Features
- **SVM-based Prediction Model**: Utilizes Support Vector Machine algorithm for high accuracy classification
- **Explainable AI**: Integrates SHAP values to provide transparency in diagnostic predictions
- **Comprehensive Medical Reporting**: Generates detailed medical reports for clinical documentation
- **User-friendly Interface**: Intuitive web interface designed for healthcare professionals
- **Cross-validation**: Ensures model reliability through rigorous validation techniques

## Dataset
The project uses the [Regensburg Pediatric Appendicitis Dataset](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis) from UCI Machine Learning Repository.

## Project Structure
```
PEDIATRIC APPENDICITIS DIAGNOSIS/
│
├── DATA/                           # Data directory
│   ├── pediatric_appendicitis_data.xlsx  # Raw dataset
│   └── processed/                  # Processed data for model training
│
├── figures/                        # Generated visualizations
│
├── models/                         # Trained models
│
├── src/                            # Source code
│   ├── analysis/                   # Data analysis scripts
│   │   ├── eda_script.py           # Exploratory Data Analysis
│   │   └── prepare_dataset.py      # Data preparation for modeling
│   │
│   ├── api/                        # API and web interface
│   │   ├── app.py                  # Flask application
│   │   ├── forms.py                # Web forms
│   │   └── templates/              # HTML templates
│   │       ├── index.html
│   │       └── results.html
│   │
│   ├── data_processing/            # Data processing utilities
│   │   └── preprocess.py           # Preprocessing functions
│   │
│   └── modeling/                   # Model training and evaluation
│       └── model_selection.py      # Model selection and training
│
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose configuration
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup script
└── README.md                       # Project documentation
```

## Installation

### Using pip
```bash
# Clone the repository
git clone <repository-url>
cd PEDIATRIC-APPENDICITIS-DIAGNOSIS

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Using Docker
```bash
# Clone the repository
git clone <repository-url>
cd PEDIATRIC-APPENDICITIS-DIAGNOSIS

# Build and run with Docker Compose
docker-compose up --build
```

## Usage

### Data Preparation and EDA
```bash
# Run exploratory data analysis
python src/analysis/eda_script.py

# Prepare the dataset for modeling
python src/analysis/prepare_dataset.py
```

### Model Training and Selection
```bash
# Train and evaluate multiple models (SVM, Random Forest, LightGBM, CatBoost)
python src/modeling/model_selection.py
```

### Run the Web Application
```bash
# Start the Flask web server
python src/api/app.py
```

## Docker Deployment

The project includes Docker configuration for easy deployment:

1. **Build the Docker image:**
   ```bash
   docker build -t pediatric-appendicitis-diagnosis .
   ```

2. **Run the Docker container:**
   ```bash
   docker run -p 5000:5000 pediatric-appendicitis-diagnosis
   ```

3. **Using Docker Compose:**
   ```bash
   docker-compose up
   ```

The web application will be available at http://localhost:5000.

## Model Performance
The SVM model has been optimized for the pediatric appendicitis diagnosis task and demonstrates:
- High sensitivity and specificity for detecting appendicitis cases
- Balanced accuracy across different patient demographics
- Robust performance with clinically relevant features
- Transparent feature importance visualization

## Recent Updates
- Converted model from RandomForest to SVM as recommended by comparative tests
- Created synthetic training data for model training and evaluation
- Added feature importance analysis using correlation with target variable
- Enhanced model verification and integration checks
- Improved medical reporting with detailed clinical recommendations

## License
This project is licensed under the MIT License - see the LICENSE file for details.