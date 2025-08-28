# 🧠 MemoryAgent

**MemoryAgent** is a memory-aware conversational workflow built using [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://www.langchain.com/).  
It enables AI agents to **extract, store, and recall user-specific memories** across interactions, making conversations more **contextual, personalized, and continuous** over time.

---

## ✨ Features

- 🔎 **Contextual Memory Retrieval** – Fetches relevant past memories using semantic search.  
- 📝 **Dynamic Memory Storage** – Automatically stores new insights with `upsert_memory`.  
- ⏳ **Temporal Awareness** – Injects current timestamps for time-sensitive reasoning.  
- 🛠 **Tool-Augmented LLM** – Lets the model call structured tools for memory updates.  
- 🔄 **Modular Workflow** – Built with `StateGraph` for extensibility and customization.  
- 🛡 **Error Resilience** – Includes robust error handling and structured logging.  

---

## 🚀 Use Cases

- **Personal AI Assistants** – Remember user preferences and conversations across sessions.  
- **Customer Support Agents** – Retain context about past issues for smoother resolutions.  
- **Knowledge Management** – Extract, organize, and recall key details from conversations.  
- **Adaptive Dialogue Systems** – Build natural, human-like conversational AI that learns over time.  

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/recallflow.git
cd recallflow
pip install -r requirements.txt
⚡ Usage

Here’s a quick example of running RecallFlow inside your project:

import asyncio
from langgraph.runtime import Runtime
from memory_agent.context import Context
from memory_agent.state import State
from recallflow import graph

async def main():
    runtime = Runtime(context=Context(user_id="user123", model="gpt-4", system_prompt="System prompt template"))
    state = State(messages=[])
    
    result = await graph.ainvoke(state, runtime=runtime)
    print(result)

asyncio.run(main())
🛠 Development

Code is organized around LangGraph workflows (StateGraph).

Memories are stored via BaseStore (customizable for databases, vector stores, or cloud backends).

LLM initialized via langchain.chat_models.init_chat_model.

Run tests:

pytest

📖 Project Structure
recallflow/
│── memory_agent/
│   ├── context.py      # Context schema (user_id, model, system_prompt)
│   ├── state.py        # State schema (conversation state, messages)
│   ├── tools.py        # Tool definitions (e.g., upsert_memory)
│   ├── utils.py        # Utility helpers
│── recallflow.py       # RecallFlow graph definition
│── requirements.txt    # Dependencies
│── README.md           # Project docs

🤝 Contributing

Contributions are welcome!
Feel free to open issues, suggest features, or submit pull requests.
