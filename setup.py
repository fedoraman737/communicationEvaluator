from setuptools import setup, find_packages

setup(
    name="communication_evaluator",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask",
        "python-dotenv",
        "openai",
        "anthropic",
        "pandas",
        "openpyxl",  # For Excel file handling
        "pytest",
    ],
    python_requires=">=3.8",
) 