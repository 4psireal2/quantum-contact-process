import subprocess
from pathlib import Path


def create_virtualenv(venv_path, python_version):
    subprocess.run(['virtualenv', '-p', python_version, str(venv_path)], check=True)


def install_requirements(venv_path):
    subprocess.run([str(venv_path / 'bin' / 'pip'), 'install', '-r', 'requirements.txt'], check=True)


def main():
    venv_name = 'tensor'
    venv_path = Path.home() / venv_name
    python_version = 'python3.11'

    print(f"Creating virtual environment '{venv_name}' with Python {python_version}...")
    create_virtualenv(venv_path, python_version)

    print("Installing packages from requirements.txt...")
    install_requirements(venv_path)

    print(f"Virtual environment '{venv_name}' setup complete.")


if __name__ == '__main__':
    main()
