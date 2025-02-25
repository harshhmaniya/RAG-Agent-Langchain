# RAG Agent Langchain

RAG-Agent-Langchain is a Retrieval-Augmented Generation (RAG) agent built using LangChain and Ollama. It integrates multiple retrieval sources - Arxiv, Wikipedia, and Langsmith documentation - to generate accurate, context-aware responses using a chat-capable LLM.

## Features

- **Multi-Source Retrieval:**  
  Combines data from Arxiv, Wikipedia, and a custom search tool for Langsmith documentation.
- **RAG Pipeline:**  
  Dynamically retrieves relevant context and generates detailed answers.
- **Document Processing:**  
  Loads and splits Langsmith documentation into manageable chunks, embedding them with Chroma.
- **LLM Integration:**  
  Powered by Ollama's `ChatOllama` (model: `llama3.2`) for natural language understanding and generation.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/harshhmaniya/RAG-Agent-Langchain.git
   cd RAG-Agent-Langchain
   ```

2. **Install Dependencies:**

   This project uses Pipenv for dependency management. Install and activate the environment with:

   ```bash
   pipenv install
   pipenv shell
   ```

## Usage

Run the main script to start the agent:

```bash
python main.py
```

The agent will process the query (e.g., "What is Attention Mechanism?") and output the answer.

## How It Works

1. **LLM Setup:**  
   Uses `ChatOllama` (model: `llama3.2`) to handle user queries.
2. **Tool Integration:**  
   - **Arxiv Tool:** Retrieves academic data.
   - **Wikipedia Tool:** Summarizes relevant information.
   - **Retrieval Tool:** Searches embedded Langsmith documentation.
3. **Agent Execution:**  
   A prompt from LangChain Hub and the integrated tools allow the agent to generate context-aware responses.

## Contributing

Contributions and suggestions are welcome! Please fork the repository and submit a pull request.

## Author

- **Name:** Harsh Mahaniya
- **[Email](harshmaniya1999@gmail.com)**
- **[LinkedIn](https://www.linkedin.com/in/harshhmaniya)**
- **[Hugging Face](https://huggingface.co/harshhmaniya)**

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
