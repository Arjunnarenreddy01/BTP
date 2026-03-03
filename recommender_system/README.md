# Dual Recommender + Agent Demo

This small Python project illustrates how you can build a dual-layer recommendation system (open electives + professors) using **dummy data** and then orchestrate it with a simple **agentic AI** pattern.

## Components

- `data.py` – dummy grades & feedback tables
- `models.py` – matrix factorization functions and scoring logic
- `agent.py` – a Python agent that calls model functions sequentially (prints reasoning steps)
- `api.py` – FastAPI server exposing the scoring/merge functions as tools
- `demo_agent_http.py` – a lightweight agent that talks to the API endpoints
- `requirements.txt` – dependencies

## Running the Demo

1. **Install dependencies** (in your workspace root):

   ```powershell
   pip install -r recommender_system/requirements.txt
   ```

2. **Run the simple agent locally** (no network):

   ```powershell
   python -m recommender_system.agent
   ```

   You'll see printed `Thought`/`Action` steps and a final ranked list.

3. **Start the FastAPI server** (for HTTP-based tools):

   ```powershell
   cd recommender_system
   uvicorn api:app --reload
   ```

   The API listens on `http://localhost:8000` with endpoints:
   * `/student/{id}` – profile
   * `/courses/{id}` – course scores
   * `/professors/{id}` – professor scores
   * `/merge` – combine scores

4. **Run the HTTP agent simulation**:

   ```powershell
   python -m recommender_system.demo_agent_http
   ```

   The script will print reasoning steps as it calls each tool.

## Extending the Demo

* Replace dummy data with a real database or CSVs
* Swap out SVD for neural CF or embeddings (see `models.build_course_embeddings(method="neural")` or initialize `RecommendationAgent(method="neural")`). The requirement file now includes `torch` and training prints loss.
* Add text-based feedback using an LLM embedding API
* Replace `demo_agent_http` with a true LLM agent (e.g. using OpenAI/Gemini via the ReAct pattern)
* Add natural‑language constraints (`"light workload"`, etc.) to the agent prompt

---

This minimal setup shows the architecture: the **models** do the heavy lifting, while the **agent** (rule‑based or LLM‑powered) coordinates tools and produces explanations. It's a springboard for your dual recommender system using agentic AI.