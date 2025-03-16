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
├── DATA/                   # Données utilisées dans le projet
│   ├── external/           # Données externes
│   ├── processed/          # Données prétraitées
│   └── raw/                # Données brutes
├── config/                 # Fichiers de configuration
├── docker/                 # Configuration Docker
├── docs/                   # Documentation
│   ├── api/                # Documentation de l'API
│   └── user_guide/         # Guide utilisateur
├── figures/                # Images et visualisations
├── models/                 # Modèles entraînés
│   ├── configs/            # Configurations des modèles
│   ├── model_history/      # Historique des modèles
│   └── trained/            # Modèles entraînés
├── notebooks/              # Notebooks Jupyter
├── outputs/                # Sorties générées par les modèles
│   └── shap_analysis/      # Analyses SHAP
├── reports/                # Rapports générés
│   ├── patients/           # Rapports pour les patients
│   └── shap_analysis/      # Rapports d'analyse SHAP
├── scripts/                # Scripts d'automatisation
├── src/                    # Code source principal
│   ├── ai_assistant/       # Module d'assistant IA (Gemini)
│   ├── analysis/           # Analyse de données
│   ├── api/                # Application web Flask
│   │   ├── static/         # Fichiers statiques
│   │   │   ├── css/        # Feuilles de style
│   │   │   ├── images/     # Images
│   │   │   ├── js/         # JavaScript
│   │   │   ├── shap_plots/ # Graphiques SHAP
│   │   │   └── temp/       # Fichiers temporaires
│   │   └── templates/      # Templates HTML
│   │       └── partials/   # Composants partiels de templates
│   ├── config/             # Configuration interne
│   ├── data/               # Gestion des données
│   ├── data_processing/    # Traitement des données
│   ├── examples/           # Exemples d'utilisation
│   ├── explainability/     # Explication des prédictions (SHAP)
│   ├── features/           # Ingénierie des caractéristiques
│   ├── modeling/           # Modélisation
│   ├── models/             # Définitions des modèles
│   ├── reporting/          # Génération de rapports
│   ├── utils/              # Utilitaires
│   ├── verification/       # Vérification des modèles
│   └── visualization/      # Visualisation des résultats
├── static/                 # Fichiers statiques globaux
├── templates/              # Templates globaux
└── tests/                  # Tests automatisés
    ├── api/                # Tests de l'API
    ├── data_processing/    # Tests de traitement des données
    ├── explainability/     # Tests d'explainabilité
    ├── integration/        # Tests d'intégration
    ├── models/             # Tests des modèles
    └── unit/               # Tests unitaires
        └── test_model.py   # Tests unitaires du modèle
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
pip install -e 
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

## Testing

### Running Tests
The project uses the Python `unittest` framework for testing. Tests are organized by module and can be run individually or all at once:

```bash
# Run all tests
python -m unittest discover

# Run tests for a specific module
python -m unittest tests.api.test_app
python -m unittest tests.api.test_routes
python -m unittest tests.data_processing.test_preprocess
```

### Coverage Analysis
For test coverage analysis, we use the `coverage` package. A utility script is provided to run coverage analysis and generate reports:

```bash
# Run coverage analysis using the utility script
python tests/run_coverage.py
```

This script will:
1. Run all unit tests with coverage analysis
2. Generate HTML and XML reports in the `/reports/coverage/` directory
3. Open the HTML report in your default browser
4. Provide a summary of coverage statistics in the terminal

### Debugging Tests
When tests fail, you can add more verbosity to see detailed output:

```bash
# Run tests with verbose output
python -m unittest discover -v
```

Common test failure reasons include:
- Missing dependencies or incorrect environment setup
- Changes to routes or function signatures
- Mock objects not properly configured
- Type errors when comparing values with mocks

## Continuous Integration

This project uses GitHub Actions for continuous integration. The workflow is configured to:
1. Run all tests on each push and pull request
2. Generate coverage reports
3. Check for code quality issues
4. Deploy automatically when changes are merged to the main branch

The CI configuration is defined in `.github/workflows/ci.yml`.

## Contributing

### Setting Up Development Environment
To set up a development environment:

```bash
# Clone the repository
git clone <repository-url>
cd PEDIATRIC-APPENDICITIS-DIAGNOSIS

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .
```

### Development Guidelines
1. Write tests for all new features
2. Maintain test coverage above 80%
3. Document all new functions, classes, and modules
4. Follow PEP 8 style guidelines
5. Submit pull requests for review before merging

## Contributors

This is a collaborative project created by:

- [Salah Eddine EL AZOUTI](https://www.linkedin.com/in/salah-eddine-el-azouti-329a50294)
- [Oussama CHEBAICHEB](https://www.linkedin.com/in/oussama-chebaicheb-963767326)
- [Anass EL HAYEL](https://www.linkedin.com/in/anass-el-hayel-0499a2346)
- [Omar Ailal](https://www.linkedin.com/in/omar-ailal-9805971a9)
- [Ilyas AOUZID](https://www.linkedin.com/in/aouzid-ilyas-76b427323)

## Recent Updates
- Converted model from RandomForest to SVM as recommended by comparative tests
- Created synthetic training data for model training and evaluation
- Added feature importance analysis using correlation with target variable
- Enhanced model verification and integration checks
- Improved medical reporting with detailed clinical recommendations

## License
This project is licensed under the MIT License - see the LICENSE file for details.