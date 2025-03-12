from setuptools import setup, find_packages

setup(
    name="pediatric_appendicitis_diagnosis",
    version="0.1.0",
    author="Salaheddine ELAZZOUTI",
    description="A machine learning system for pediatric appendicitis diagnosis",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "lightgbm>=4.0.0",
        "catboost>=1.0.0",
        "plotly>=5.0.0",
        "flask>=2.0.0",
        "flask-wtf>=1.0.0",
        "waitress>=2.0.0",
        "joblib>=1.1.0",
        "tqdm>=4.62.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)