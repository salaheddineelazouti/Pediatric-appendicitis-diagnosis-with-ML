"""
Test runner for the Pediatric Appendicitis Diagnosis project.
This script discovers and runs all tests in the tests directory.
"""

import os
import sys
import unittest

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_tests():
    """Discover and run all tests in the tests directory."""
    # Discover all test modules
    test_suite = unittest.defaultTestLoader.discover(
        start_dir=os.path.dirname(os.path.abspath(__file__)),
        pattern='test_*.py'
    )
    
    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    result = test_runner.run(test_suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    # Allow running specific test modules
    if len(sys.argv) > 1:
        # If specific test modules are provided, run them
        specific_tests = []
        for arg in sys.argv[1:]:
            if arg.endswith('.py'):
                # If the full path is provided
                specific_tests.append(arg)
            else:
                # If just the module name is provided
                specific_tests.append(f"test_{arg}.py")
        
        test_suite = unittest.TestSuite()
        for test_file in specific_tests:
            # Convert path to module name
            module_name = os.path.splitext(os.path.basename(test_file))[0]
            # Try to find the module in different test directories
            for root, _, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
                if f"{module_name}.py" in files:
                    # Get the relative path from the project root
                    rel_path = os.path.relpath(root, project_root)
                    # Convert path to module path
                    module_path = rel_path.replace(os.sep, '.') + '.' + module_name
                    # Add to test suite
                    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromName(module_path))
        
        # Run specific tests
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(test_suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        # Run all tests
        sys.exit(run_tests())
