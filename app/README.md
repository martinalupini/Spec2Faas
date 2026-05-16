# Application

This folder contains the main application code.

To start the application, run:

- `main.py` for the standard version
- `main_RAG.py` to enable the RAG mechanism

The LLM to use can be selected in the `config.yaml` file located in the root directory.

## Requirements

Before running the application, create a `.env` file containing:

```env
GEMINI_API_KEY=your_api_key

SERVERLEDGE_URL=your_serverledge_url
SERVERLEDGE_USERNAME=your_serverledge_username
SERVERLEDGE_PASS=your_serverledge_password

UI=False
```

Make sure that Ollama is installed before starting the application.