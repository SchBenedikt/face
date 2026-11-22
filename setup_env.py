"""
Environment setup module for face recognition
Handles environment configuration and prevents threading issues
"""
import os
import sys
import warnings
import multiprocessing


def setup_environment():
    """
    Set up the environment for optimal face recognition performance
    """
    try:
        # Set threading and parallelization settings
        # Keep safe defaults, but let operators tune CPU parallelism before
        # startup instead of silently overwriting their performance settings.
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        
        # Set multiprocessing start method for compatibility
        if hasattr(multiprocessing, 'set_start_method'):
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
        
        # Suppress common warnings for cleaner output
        warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
        warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
        warnings.filterwarnings('ignore', message="pkg_resources is deprecated as an API.*")
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        # Do not disable an installed GPU runtime by default. DeepFace can use
        # ONNX Runtime, TensorFlow, or PyTorch depending on the selected
        # backend and installed packages. CPU-only mode remains available for
        # machines with unsupported drivers or for reproducible debugging.
        force_cpu = os.environ.get('FACE_FORCE_CPU', '').strip().lower()
        if force_cpu in {'1', 'true', 'yes', 'on'}:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Optimize for CPU performance
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
        
        return True
        
    except Exception as e:
        print(f"Warning: Could not fully setup environment: {e}")
        return False


def get_optimal_workers():
    """
    Get optimal number of workers for the current system
    """
    try:
        cpu_count = multiprocessing.cpu_count()
        # Use 75% of available cores, minimum 1, maximum 8 for face recognition
        optimal = max(1, min(8, int(cpu_count * 0.75)))
        return optimal
    except:
        return 4  # Default fallback


def get_optimal_batch_size():
    """
    Get optimal batch size based on system memory and CPU cores
    """
    try:
        cpu_count = multiprocessing.cpu_count()
        # Increase batch size based on available cores
        if cpu_count >= 12:
            return 50
        elif cpu_count >= 8:
            return 30
        elif cpu_count >= 4:
            return 20
        else:
            return 10
    except:
        return 10  # Default fallback


if __name__ == "__main__":
    print("Setting up environment...")
    success = setup_environment()
    print(f"Environment setup: {'Success' if success else 'Partial'}")
    print(f"Optimal workers: {get_optimal_workers()}")
    print(f"Optimal batch size: {get_optimal_batch_size()}")
