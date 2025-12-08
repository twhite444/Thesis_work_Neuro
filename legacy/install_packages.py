import subprocess
import sys

def install(package):
    """ Install packages using pip quietly. """
    try:
        # Using the --quiet flag to reduce output verbosity
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, '--quiet'])
        print(f"{package} installed successfully.")
    except Exception as e:
        print(f"Failed to install {package}: {e}")

def main():
    packages = [
        'pandas',          # For data manipulation
        'numpy',           # For numerical operations
        'scikit-learn',    # For machine learning algorithms
        'rdkit-pypi',      # For cheminformatics
        'matplotlib',      # For plotting graphs
        'seaborn',         # For more statistical visualizations
        'pyrfume',         # For olfactory research (if available)
        'keras',           # For deep learning models
        'tensorflow',      # Backend for Keras
        'scipy',           # For scientific computing and technical computing
        'torch',           # PyTorch for deep learning models, CPU version
        'torchvision'      # For dealing with image data in PyTorch
    ]

    for package in packages:
        install(package)

if __name__ == "__main__":
    main()
