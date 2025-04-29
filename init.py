#!/usr/bin/env python3
"""
PyTorch Environment Setup Script
This script sets up a Python virtual environment and installs PyTorch with CUDA support.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

# Environment path - global constant
VENV_PATH = "pytorch_env"

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def is_windows():
    return platform.system() == "Windows"

def is_linux():
    return platform.system() == "Linux"

def is_mac():
    return platform.system() == "Darwin"

def print_step(message):
    print(f"{Colors.HEADER}[SETUP]{Colors.ENDC} {message}")

def print_error(message):
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")

def print_success(message):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {message}")

def print_command(message):
    print(f"{Colors.BLUE}[COMMAND]{Colors.ENDC} {message}")

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def check_python_version():
    print_step("Checking Python version...")
    major, minor, _ = platform.python_version_tuple()
    major, minor = int(major), int(minor)
    
    if major == 3 and minor >= 9 and minor <= 12:
        print_success(f"Python {major}.{minor} is compatible with PyTorch.")
        return True
    else:
        print_error(f"Python {major}.{minor} may not be fully compatible with PyTorch.")
        print_error("Recommended versions are Python 3.9-3.12.")
        return False

def create_venv():
    print_step(f"Creating virtual environment at {VENV_PATH}...")
    
    if os.path.exists(VENV_PATH):
        print_step(f"Virtual environment already exists at {VENV_PATH}")
        return True
    
    python_cmd = sys.executable
    result = run_command(f"{python_cmd} -m venv {VENV_PATH}")
    
    if result:
        print_success(f"Virtual environment created at {VENV_PATH}")
        return True
    else:
        print_error("Failed to create virtual environment")
        return False

def get_activation_command():
    if is_windows():
        activate_path = os.path.join(VENV_PATH, "Scripts", "Activate.ps1")
        return f"powershell -ExecutionPolicy Bypass -File {activate_path}"
    elif is_linux() or is_mac():
        activate_path = os.path.join(VENV_PATH, "bin", "activate")
        return f"source {activate_path}"
    else:
        print_error("Unsupported operating system")
        return None

def get_install_commands():
    commands = []
    # PyTorch installation command
    commands.append("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    # Check if requirements.txt exists
    if os.path.exists("requirements.txt"):
        commands.append("pip install -r requirements.txt")
        
    return commands

def main():
    print_step("Starting PyTorch environment setup...")
    
    if not check_python_version():
        proceed = input("Continue anyway? (y/n): ").lower()
        if proceed != 'y':
            sys.exit(1)
    
    if not create_venv():
        sys.exit(1)
    
    activation_cmd = get_activation_command()
    if not activation_cmd:
        sys.exit(1)
    
    install_cmds = get_install_commands()
    
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}NEXT STEPS:{Colors.ENDC}")
    print("=" * 80)
    
    print("1. Activate the virtual environment with the following command:")
    print_command(activation_cmd)
    
    print("\n2. Install PyTorch and dependencies:")
    for i, cmd in enumerate(install_cmds):
        print_command(cmd)
        
    if len(install_cmds) > 1:
        print_success("Found requirements.txt - it will be installed automatically")
    
    # Create activation shortcuts
    if is_windows():
        with open("activate_pytorch.bat", "w") as f:
            f.write(f"@echo off\necho Activating PyTorch environment...\n{VENV_PATH}\\Scripts\\activate.bat\n")
        print("\nYou can also use the created 'activate_pytorch.bat' file to activate the environment.")
    
    elif is_linux() or is_mac():
        with open("activate_pytorch.sh", "w") as f:
            f.write(f"#!/bin/bash\necho Activating PyTorch environment...\nsource {VENV_PATH}/bin/activate\n")
        os.chmod("activate_pytorch.sh", 0o755)
        print("\nYou can also use the created 'activate_pytorch.sh' file to activate the environment:")
        print_command("source ./activate_pytorch.sh")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()