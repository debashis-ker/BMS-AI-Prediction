# BMS AI

A brief description of what this AI project does and its primary goals.

***

## Getting Started

This guide will walk you through setting up the project on your local machine for development and testing purposes.

***

## Prerequisites

Before you begin, ensure you have the **`uv`** Python package manager installed.

1.  **Check if `uv` is installed** by opening your terminal and running:
    ```bash
    uv --version
    ```
2.  **If `uv` is not installed**, run the following command in **PowerShell**:
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
    > **Note:** You may need to restart your terminal after installation for the `uv` command to be recognized.

***

## Installation Guide

Follow these steps to get your development environment running.

1.  **Create a Virtual Environment**
    Navigate to the project's root directory and create a Python 3.11 virtual environment.
    ```bash
    uv venv -p 3.11
    ```

2.  **Activate the Environment**
    Activate the newly created environment. The command depends on your terminal.
    * **Git Bash / WSL / Linux / macOS:**
        ```bash
        source .venv/bin/activate
        ```
    * **Windows PowerShell:**
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
    > You'll know it's active when your command prompt is prefixed with `(.venv)`.

3.  **Install Core Dependencies**
    With the environment activated, install the project's required packages.
    ```bash
    uv pip install -e .
    ```

***