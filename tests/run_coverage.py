"""
Coverage analysis script for the Pediatric Appendicitis Diagnosis project.
This script runs the tests with coverage analysis to identify untested code.
"""

import os
import sys
import subprocess
import webbrowser
from datetime import datetime

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_coverage():
    """Run tests with coverage and generate a report."""
    # Define output directories
    reports_dir = os.path.join(project_root, 'reports', 'coverage')
    html_dir = os.path.join(reports_dir, 'html')
    
    # Create directories if they don't exist
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)
    
    # Define the source directories to analyze
    source_dirs = [
        'src/api',
        'src/data_processing',
        'src/explainability',
        'src/ai_assistant'
    ]
    
    # Format source directories for coverage command
    source_arg = ','.join(source_dirs)
    
    # Generate timestamp for report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Define report paths
    html_report = os.path.join(html_dir, f'coverage_report_{timestamp}')
    xml_report = os.path.join(reports_dir, f'coverage_report_{timestamp}.xml')
    
    try:
        # Run coverage for all tests
        print("Running tests with coverage analysis...")
        
        # Erase any previous coverage data
        subprocess.run([sys.executable, '-m', 'coverage', 'erase'], check=True)
        
        # Run tests with coverage
        subprocess.run([
            sys.executable, '-m', 'coverage', 'run',
            '--source', source_arg,
            '-m', 'unittest', 'discover',
            '-s', os.path.dirname(os.path.abspath(__file__)),
            '-p', 'test_*.py'
        ], check=True)
        
        # Generate HTML report
        print(f"Generating HTML coverage report at {html_report}...")
        subprocess.run([
            sys.executable, '-m', 'coverage', 'html',
            '-d', html_report
        ], check=True)
        
        # Generate XML report for potential CI integration
        print(f"Generating XML coverage report at {xml_report}...")
        subprocess.run([
            sys.executable, '-m', 'coverage', 'xml',
            '-o', xml_report
        ], check=True)
        
        # Print coverage report to console
        print("\nCoverage Summary:")
        subprocess.run([sys.executable, '-m', 'coverage', 'report'], check=True)
        
        # Open HTML report in browser
        index_file = os.path.join(html_report, 'index.html')
        if os.path.exists(index_file):
            print(f"\nOpening HTML report in browser: {index_file}")
            webbrowser.open(f"file://{index_file}")
        
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running coverage: {e}")
        return 1

if __name__ == "__main__":
    # Check if coverage is installed
    try:
        subprocess.run([sys.executable, '-m', 'coverage', '--version'], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        print("Coverage.py is not installed. Installing now...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'coverage'], check=True)
            print("Coverage.py installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install coverage.py: {e}")
            sys.exit(1)
    
    # Run coverage analysis
    sys.exit(run_coverage())
