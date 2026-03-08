import os
import sys
import subprocess
import platform

# Check Python version
def check_python():
    version = sys.version_info
    print(f"Python Version : {sys.version}")
    print(f"Platform       : {platform.system()} {platform.release()}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Python 3.8+ is required.")
        sys.exit(1)
    print("Python version OK\n")

# Project folder structure
PROJECT_STRUCTURE = {
    "exam_anxiety_detector": [
        "data/raw", "data/processed",
        "model/saved_model",
        "backend", "frontend",
        "notebooks", "logs", "tests",
    ]
}

# Create all project folders
def create_project_folders():
    root = list(PROJECT_STRUCTURE.keys())[0]
    for subfolder in PROJECT_STRUCTURE[root]:
        path = os.path.join(root, subfolder)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")
    print("All folders created\n")
    return root

# Create virtual environment
def setup_virtual_env(project_root):
    venv_path = os.path.join(project_root, "venv")
    if not os.path.exists(venv_path):
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        print(f"Virtual environment created at: {venv_path}")
    else:
        print(f"Virtual environment already exists at: {venv_path}")

    # Show activation command based on OS
    if platform.system() == "Windows":
        activate = f"{venv_path}\\Scripts\\activate"
    else:
        activate = f"source {venv_path}/bin/activate"
    print(f"Activate with: {activate}\n")

# Required packages list
REQUIREMENTS = [
    "torch>=2.0.0",
    "transformers>=4.39.0",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "streamlit>=1.33.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "pydantic>=2.0.0",
    "requests>=2.31.0",
    "plotly>=5.18.0",
    "python-dotenv>=1.0.0",
]

# Write requirements.txt
def create_requirements_file(project_root):
    req_path = os.path.join(project_root, "requirements.txt")
    with open(req_path, "w") as f:
        f.write("# Exam Anxiety Detector Dependencies\n\n")
        for pkg in REQUIREMENTS:
            f.write(pkg + "\n")
    print(f"requirements.txt saved at: {req_path}")
    print("Install with: pip install -r requirements.txt\n")

# Files to create inside project
INIT_FILES = {
    "backend/__init__.py": "",
    "backend/config.py": (
        "MODEL_DIR = '../model/saved_model'\n"
        "MAX_LEN = 128\n"
        "HOST = '0.0.0.0'\n"
        "PORT = 8000\n"
    ),
    "frontend/__init__.py": "",
    "tests/__init__.py": "",
}

# Create initial source files
def create_init_files(project_root):
    for rel_path, content in INIT_FILES.items():
        full_path = os.path.join(project_root, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        print(f"Created: {full_path}")
    print("Project structure ready\n")

# .gitignore content
GITIGNORE = """__pycache__/
*.pyc
venv/
.env
model/saved_model/*.bin
model/saved_model/*.safetensors
data/raw/
logs/
.vscode/
.idea/
*.log
"""

# Initialize git and write .gitignore
def init_git(project_root):
    git_dir = os.path.join(project_root, ".git")
    if not os.path.exists(git_dir):
        result = subprocess.run(["git", "init", project_root], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Git initialized at: {project_root}")
        else:
            print(f"Git not found: {result.stderr.strip()}")
    else:
        print("Git repo already exists")

    gitignore_path = os.path.join(project_root, ".gitignore")
    with open(gitignore_path, "w") as f:
        f.write(GITIGNORE)
    print(f".gitignore created\n")


if __name__ == "__main__":
    print("MILESTONE 1 - Environment Setup\n")
    check_python()
    root = create_project_folders()
    setup_virtual_env(root)
    create_requirements_file(root)
    create_init_files(root)
    init_git(root)
    print("MILESTONE 1 COMPLETE")
