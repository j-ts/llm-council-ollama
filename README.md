# LLM Council

![llmcouncil](header.jpg)

<!-- Table of Contents -->
- [Vibe Code Alert](#-vibe-code-alert)
- [New Features (Recent Changes)](#-new-features-recent-changes)
- [Setup](#-setup)
- [Running the Application](#-running-the-application)
- [Tech Stack](#-tech-stack)
- [License](#-license)

Instead of asking a question to a single LLM, this app lets you assemble your own **LLM Council** from multiple models—whether local (via **Ollama**), cloud-based (**Ollama**, **OpenRouter**), or custom endpoints (**OpenAI-compatible** APIs). The web interface sends your query to all council members simultaneously, collects their responses, then has them review and rank each other's answers anonymously. Finally, a designated **Chairman** model synthesizes the ranked outputs into a single, polished response.

In a bit more detail, here is what happens when you submit a query:

1. **Stage 1: First opinions** – The user query is sent to all LLMs individually and the responses are collected. Each response is shown in a tab view for easy inspection.
2. **Stage 2: Review** – Every LLM receives the other models' responses (identities are anonymised) and is asked to rank them on accuracy and insight.
3. **Stage 3: Final response** – The designated Chairman model aggregates the ranked outputs into a single, polished answer for the user.

---

## ✨ Vibe Code Alert

This project was 99% *vibe‑coded* as a fun Saturday hack while exploring side‑by‑side LLM comparisons (see the original tweet [here](https://x.com/karpathy/status/1990577951671509438)). The code is intentionally lightweight and may contain shortcuts. **It is provided as‑is for inspiration; no ongoing support is guaranteed.**

---

## 🎨 New Features (Recent Changes)

- **API Keys via Environment Variables** – You can now store model credentials outside `config.json`. Each API key field supports a `Direct | Env Var` toggle so you can decide per model; selecting Env Var saves values as `env:YOUR_VAR_NAME` and the backend resolves the secret from the environment at runtime.
- **General Setting for Defaults** – In Settings → Other Settings, a new “Store API Key as ENV variable, not in json” checkbox controls the default mode when adding future models (existing entries stay untouched). Even with the box unchecked you can still opt into env vars per model via the toggle.
- **Masked Config Responses** – All config/API responses now mask plain API keys and preserve `env:` references so the UI can show which environment variable is referenced without revealing the actual secret.
- **Model Registry** – Complete redesign of model configuration! Now you define individual model instances (e.g., "My Local Llama", "GPT-4 via OpenRouter") with their own credentials in a central registry, then select from these pre-configured models for your Council and Chairman.
- **Multi-Provider Support** – Supports **Local Ollama**, **OpenRouter**, and **OpenAI-Compatible** endpoints (e.g., LM Studio, vLLM).
- **Settings UI Redesign** – Three-tab interface:
  - **Models Tab**: Add, edit, and delete model configurations with labels and type badges
  - **Ollama Settings Tab**: Configure global Ollama parameters (context window, serialization)
  - **Council Configuration Tab**: Select models from your registry for Council and Chairman
- **Automated Setup Scripts** – New `setup.sh` and `setup.bat` scripts automate dependency installation and Redis setup for faster onboarding.
- **Dark Theme** – A sleek dark UI is now the default.
- **Frontend Improvements** – Fixed tab naming display issues in peer rankings and evaluations views.
- **Docker Setup** – A minimal Dockerfile and compose script for quick containerised deployment.

---

## 🚀 Setup

### Quick Start Guide

Choose the setup method that works best for you:

- **🐳 Docker (Recommended)** – Easiest option, everything containerized. Best for quick demos and production.
- **⚡ Automated Scripts** – One-command setup with `./setup.sh` (macOS/Linux) or `setup.bat` (Windows). Handles dependencies and Redis automatically.
- **🛠️ Manual Installation** – Full control over each component. Best for development and customization.

> [!TIP]
> New to the project? Start with Docker or the automated setup scripts for the fastest experience.

### Option 1: Docker Deployment (Recommended)

**Prerequisites:**
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system

**All Operating Systems (macOS / Linux / Windows):**

1. Clone the repository and navigate to the project directory
2. Create a `.env` file in the project root (optional, only if using OpenRouter):
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-...
   REDIS_HOST=redis
   REDIS_PORT=6379
   ```
   Get your API key at [openrouter.ai](https://openrouter.ai/).

3. Build and run with Docker Compose:
   ```bash
   docker compose up --build
   ```

4. Open `http://localhost:5173` in your browser

This brings up Redis, the backend, the background worker, and the frontend—all configured and ready to use.

---

### Option 2: Automated Setup Scripts

Use the provided setup scripts to automatically install dependencies and configure Redis:

**macOS / Linux:**
```bash
./setup.sh
```

**Windows:**
```powershell
.\setup.bat
```

These scripts will:
- ✓ Check for required dependencies (Python, Node.js, Docker)
- ✓ Install Python and frontend dependencies
- ✓ Create a Redis container via Docker
- ✓ Generate a `.env` file from the template

**After setup completes:**
```bash
# macOS / Linux
./start-background.sh

# Windows
.\start-background.bat
```

Then open `http://localhost:5173` in your browser.

---

### Option 3: Manual Installation

If you prefer not to use Docker, follow these OS-specific instructions:

#### Prerequisites (All OS)
- **Python 3.10+** with [uv](https://docs.astral.sh/uv/) or pip
- **Node.js 16+** and npm
- **Redis** server

---

#### macOS

**1. Install Dependencies**

Redis:
```bash
brew install redis
```

Python dependencies:
```bash
uv sync
# or
pip install -r requirements.txt
```

Frontend:
```bash
cd frontend
npm install
cd ..
```

**2. Configure Environment**

Create a `.env` file:
```bash
OPENROUTER_API_KEY=sk-or-v1-...
REDIS_HOST=localhost
REDIS_PORT=6380
```

**3. Start Services**

Start Redis:
```bash
redis-server --port 6380
```

In separate terminal windows, start:

Background worker:
```bash
uv run rq worker council --url redis://localhost:6380/0
```

Backend:
```bash
uv run python -m backend.main
```

Frontend:
```bash
cd frontend
npm run dev
```

**4. Access the App**

Open `http://localhost:5173` in your browser.

---

#### Linux

**1. Install Dependencies**

Redis (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install redis-server
```

Redis (Fedora/RHEL):
```bash
sudo dnf install redis
```

Python dependencies:
```bash
uv sync
# or
pip install -r requirements.txt
```

Frontend:
```bash
cd frontend
npm install
cd ..
```

**2. Configure Environment**

Create a `.env` file:
```bash
OPENROUTER_API_KEY=sk-or-v1-...
REDIS_HOST=localhost
REDIS_PORT=6380
```

**3. Start Services**

Start Redis:
```bash
redis-server --port 6380
```

In separate terminal windows, start:

Background worker:
```bash
uv run rq worker council --url redis://localhost:6380/0
```

Backend:
```bash
uv run python -m backend.main
```

Frontend:
```bash
cd frontend
npm run dev
```

**4. Access the App**

Open `http://localhost:5173` in your browser.

---

#### Windows

**1. Install Dependencies**

Redis (using [Memurai](https://www.memurai.com/) or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)):
- **Option A**: Download and install [Memurai](https://www.memurai.com/get-memurai) (Redis for Windows)
- **Option B**: Use WSL and follow the Linux instructions above

Python dependencies (PowerShell):
```powershell
uv sync
# or
pip install -r requirements.txt
```

Frontend:
```powershell
cd frontend
npm install
cd ..
```

**2. Configure Environment**

Create a `.env` file:
```
OPENROUTER_API_KEY=sk-or-v1-...
REDIS_HOST=localhost
REDIS_PORT=6380
```

**3. Start Services**

Start Redis (Memurai or WSL):
```powershell
# If using Memurai, it runs as a service automatically
# If using WSL: wsl redis-server --port 6380
```

In separate PowerShell windows, start:

Background worker:
```powershell
uv run rq worker council --url redis://localhost:6380/0
```

Backend:
```powershell
uv run python -m backend.main
```

Frontend:
```powershell
cd frontend
npm run dev
```

**4. Access the App**

Open `http://localhost:5173` in your browser.

---

### Configure Models via UI

After starting the app, configure your models using the **Model Registry**:

#### Adding Models
1. Click the ⚙️ Settings icon in the sidebar
2. Go to the **Models** tab
3. Click **+ Add Model**
4. Fill in the model details:
   - **Label**: A friendly name (e.g., "My Local Llama")
   - **Type**: Choose Ollama, OpenRouter, or OpenAI Compatible
   - **Model Name**: The actual model identifier (e.g., `llama3`, `openai/gpt-4o`)
   - **Base URL**: For Ollama and OpenAI Compatible (e.g., `http://localhost:11434`)
   - **API Key**: For OpenRouter and OpenAI Compatible
5. Click **Save Model**

#### Configuring Ollama Settings
Global Ollama settings are now in a dedicated tab:
- **Context Window (num_ctx)**: Default context size for all Ollama models (e.g., 4096, 8192)
- **Serialize Requests**: Run Ollama models sequentially to avoid GPU thrashing

#### Selecting Council Models and Chairman
1. Go to the **Council Configuration** tab
2. Check the boxes for models you want in your Council
3. Select a Chairman model from the dropdown
4. Click **Save Changes**

**Local Ollama Setup:**
1. Ensure Ollama is running (`ollama serve`)
2. Pull models you want to use (e.g., `ollama pull mistral`, `ollama pull llama3`)
3. In Settings → Models tab, add each model with:
   - Type: Ollama
   - Base URL: `http://localhost:11434`
   - Model Name: The model you pulled (e.g., `mistral`)

**Default Configuration:**
The app starts with free OpenRouter models configured. You can add your own models or modify these in the Settings UI.

---

## 🧩 Supported Models

The application can work with the following model providers:

- **Ollama** – Run local models via the Ollama server (e.g., `http://localhost:11434`). Ideal for offline use.
- **OpenRouter** – Cloud‑based models with a free tier. Requires an API key set in `.env` (`OPENROUTER_API_KEY`).
- **OpenAI‑compatible** – Any OpenAI‑style endpoint such as LM Studio, vLLM, or custom deployments. Configure the base URL and API key in the Model Registry.

Configure these models in the **Models** tab of the Settings UI. See the **Setup** section for details on adding each type.

---

## ⚙️ Configuration Format

Models are now stored in a **Model Registry** in `data/config.json`:

```json
{
  "models": {
    "unique-model-id": {
      "label": "My Model Label",
      "type": "ollama|openrouter|openai-compatible",
      "model_name": "actual-model-name",
      "base_url": "http://localhost:11434",
      "api_key": "sk-..."
    }
  },
  "ollama_settings": {
    "num_ctx": 4096,
    "serialize_requests": false
  },
  "council_models": ["model-id-1", "model-id-2"],
  "chairman_model": "model-id-3"
}
```

**Key Features:**
- Each model has its own credentials and settings
- Models are referenced by ID throughout the app
- Global Ollama settings apply to all Ollama models
- Automatic migration from old configuration format

---

## 🔧 Troubleshooting

### Common Issues

**Redis Connection Errors**
```
rq.exceptions.ConnectionError: Error while reading from socket
```
- **Solution**: Ensure Redis is running on the correct port (default: 6380)
- Check Docker: `docker ps | grep redis`
- Restart Redis: `docker restart llm-council-redis`

**Ollama Service Not Reachable**
```
Unable to reach Ollama at http://localhost:11434
```
- **Solution**: Start Ollama service: `ollama serve`
- Verify models are pulled: `ollama list`
- If using Docker, use `http://host.docker.internal:11434` as the base URL

**OpenRouter API Errors**
```
401 Unauthorized or Invalid API Key
```
- **Solution**: Check your `.env` file has the correct `OPENROUTER_API_KEY`
- Get a new key at [openrouter.ai/keys](https://openrouter.ai/keys)
- Restart the backend after updating `.env`

**Port Already in Use**
```
Address already in use: 5173, 8010, or 6380
```
- **Solution**: Change the port in the respective config:
  - Frontend (5173): Edit `frontend/vite.config.js`
  - Backend (8010): Edit `backend/main.py` or use env var
  - Redis (6380): Update `REDIS_PORT` in `.env`

**Frontend Tab Names Showing "(OpenRouter)" Repeatedly**
- **Solution**: This was a known bug that has been fixed in recent updates
- Update to the latest version: `git pull origin main`
- Clear browser cache and reload

**Docker Networking Issues (localhost vs host.docker.internal)**
- **Solution**: The app automatically rewrites URLs when running in Docker:
  - `localhost` → `host.docker.internal` (when in container)
  - `host.docker.internal` → `127.0.0.1` (when on host)
- For manual override, set the correct base URL in Settings → Models

### Getting Help

If you encounter other issues:
1. Check the backend logs in your terminal
2. Check browser console for frontend errors
3. Review the [GitHub Issues](https://github.com/USERNAME/llm-council/issues)
4. Open a new issue with:
   - Your OS and setup method (Docker/Scripts/Manual)
   - Error messages and logs
   - Steps to reproduce

---

## 🛠️ Tech Stack

- **Backend:** FastAPI (Python 3.10+), async httpx, Redis + RQ for job queuing
- **Frontend:** React + Vite, react‑markdown for rendering
- **Storage:** JSON files in `data/` for config and conversations
- **Package Management:** uv for Python, npm for JavaScript

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. Please ensure that any new code follows the existing style and includes appropriate documentation.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
