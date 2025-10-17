# BMS AI



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

***

## Installation Guide

Follow these steps to get your development environment running.

1.  **Create a Virtual Environment**
    Navigate to the project's root directory and create a Python 3.11 virtual environment.
    ```bash
    uv venv -p 3.10
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

3.  **Install Core Dependencies**
    With the environment activated, install the project's required packages.
    ```bash
    uv pip install -r requirements.txt
    ```

***

## Configuration

- `PORT`: The port the server will run on. Defaults to `8000`.
- `WORKERS`: The number of worker processes for production. Defaults to `4`.
- `ENVIRONMENT`: Set to `development` for reload mode or `production` for multi-worker mode. Defaults to `development`.



## Running the Server

To run the server, navigate to the project root directory and use:

```bash
# For development (with auto-reload)
python -m src.bms_ai.api.server

# For production (with multiple workers, no reload)
python -m src.bms_ai.api.server --prod
```

Or set `ENVIRONMENT=production` in your `.env` file.

***

## API Endpoints

### Utilities

#### POST `/utils/fetch-instances`
Fetch process instances from IKON service.

**Required Environment Variables:**
- `IKON_BASE_URL`: Base URL for the IKON service
- `SOFTWARE_ID`: Software identifier for IKON service (can be overridden in request body)

**Request Body:**
```json
{
  "ticket": "auth_ticket_string",
  "process_name": "process_name",
  "account_id": "account_id",
  "software_id": "optional_software_id",
  "predefined_filters": {},
  "process_variable_filters": {},
  "task_variable_filters": {},
  "mongo_where_clause": "optional_query",
  "projections": ["field1", "field2"],
  "all_instances": false
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/utils/fetch-instances" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket": "your_ticket",
    "process_name": "YourProcessName",
    "account_id": "your_account_id"
  }'
```

***