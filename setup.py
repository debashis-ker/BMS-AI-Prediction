from setuptools import setup, find_packages # type: ignore

"""
For future use

import sys
import ast
import os

def find_imports_in_dir(src_dir):
    
    ## Scan all .py files in src_dir and return a set of top-level imported packages.
    
    imports = set()

    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        node = ast.parse(f.read(), filename=filepath)
                    except SyntaxError:
                        continue  # skip files with syntax errors (rare)

                    for n in ast.walk(node):
                        # Handle "import x"
                        if isinstance(n, ast.Import):
                            for alias in n.names:
                                imports.add(alias.name.split('.')[0])  # only top-level pkg
                        # Handle "from x import y"
                        elif isinstance(n, ast.ImportFrom):
                            if n.module:  # avoid "from . import x"
                                imports.add(n.module.split('.')[0])

    return imports




def filter_external_packages(imports):
    if hasattr(sys, "stdlib_module_names"):  # Python 3.10+
        stdlib = sys.stdlib_module_names
    else:
        # fallback: minimal stdlib set
        stdlib = {"os", "sys", "math", "re", "json", "logging"}
    return sorted(pkg for pkg in imports if pkg not in stdlib)

"""

setup(
    name='bms_ai',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'xgboost',
        'seaborn',
        'matplotlib',
        'seaborn',
        #'tensorflow',
        #'keras-tuner',
        'joblib',
        'gunicorn',
        'pydantic',
        'fastapi',
        'uvicorn[standard]',
        'python-dotenv'
    ],
    # install_requires=filter_external_packages(find_imports_in_dir("src"))
    author='Debashis Mondal',
    author_email='debashis.mondal@keross.com',
    description='',
    url='',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)