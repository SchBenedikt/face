"""
Environment setup module to prevent threading issues
"""
import os
import warnings

def setup_environment():
    """Set up environment variables to prevent threading issues and suppress warnings"""
    
    # Prevent TensorFlow threading issues
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Disable telemetry
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    print("Environment configured to prevent threading issues and telemetry")