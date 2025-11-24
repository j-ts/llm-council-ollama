# LLM Council

![llmcouncil](header.jpg)

<!-- Table of Contents -->
- [Vibe Code Alert](#-vibe-code-alert)
- [New Features (Recent Changes)](#-new-features-recent-changes)
- [Setup](#-setup)
- [Running the Application](#-running-the-application)
- [Tech Stack](#-tech-stack)
- [License](#-license)

The idea of this repo is that instead of asking a question to your favorite LLM provider (e.g. OpenAI GPT 5.1, Google Gemini 3.0 Pro, Anthropic Claude Sonnet 4.5, xAI Grok 4, eg.c), you can group them into your "LLM Council". This repo is a simple, local web app that essentially looks like ChatGPT except it uses OpenRouter to send your query to multiple LLMs, it then asks them to review and rank each other's work, and finally a Chairman LLM produces the final response.

In a bit more detail, here is what happens when you submit a query:

1. **Stage 1: First opinions** – The user query is sent to all LLMs individually and the responses are collected. Each response is shown in a tab view for easy inspection.
2. **Stage 2: Review** – Every LLM receives the other models' responses (identities are anonymised) and is asked to rank them on accuracy and insight.
3. **Stage 3: Final response** – The designated Chairman model aggregates the ranked outputs into a single, polished answer for the user.

---

## ✨ Vibe Code Alert

This project was 99% *vibe‑coded* as a fun Saturday hack while exploring side‑by‑side LLM comparisons (see the original tweet [here](https://x.com/karpathy/status/1990577951671509438)). The code is intentionally lightweight and may contain shortcuts. **It is provided as‑is for inspiration; no ongoing support is guaranteed.**

---

## 🎨 New Features (Recent Changes)

- **Multi-Provider Support** – Now supports **Local Ollama**, **OpenRouter**, and **Direct OpenAI/Anthropic** connections.
- **Settings UI** – Configure providers, API keys, and model selection directly from the UI.
- **Dark Theme** – A sleek dark UI is now the default.
- **Sidebar Background Fix** – The sidebar now respects the dark theme, fixing the white background issue.
- **Free OpenRouter Models** – The default configuration now uses free‑tier OpenRouter models, lowering the barrier to try the app out of the box.
- **Docker Setup** – A minimal Dockerfile and compose script have been added for quick containerised deployment.

![dark theme screenshot](dark_theme.png)

---

## 🚀 Setup

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for Python dependency management.

**Backend:**
```bash
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 2. Configure API Key
Create a `.env` file in the project root:
```bash
OPENROUTER_API_KEY=sk-or-v1-...
```
Get your API key at [openrouter.ai](https://openrouter.ai/).

### 3. Configure Models
You can now configure models directly in the application UI!
1. Click "Settings" in the sidebar.
2. Select your provider (OpenRouter, Ollama, or OpenAI Compatible).
3. Enter API keys or Base URLs.
4. Add/Remove models for the Council and select a Chairman.

**Default Configuration:**
The app starts with a default configuration using free OpenRouter models. You can override this in the UI.

**Local Ollama Setup:**
1. Ensure Ollama is running (`ollama serve`).
2. Pull models you want to use (e.g., `ollama pull mistral`, `ollama pull llama3`).
3. In the app Settings, select "Ollama (Local)".
4. The Base URL defaults to `http://localhost:11434`.
5. The "Council Models" dropdown will populate with your local models.

### 4. (Optional) Docker Deployment
A simple Dockerfile is provided. To build and run:
```bash
docker build -t llm-council .
docker run -p 5173:5173 -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY llm-council
```

---

## ▶️ Running the Application

**Option 1: Use the start script**
```bash
./start.sh
```

**Option 2: Run manually**

Backend:
```bash
uv run python -m backend.main
```
Frontend:
```bash
cd frontend
npm run dev
```
Then open http://localhost:5173 in your browser.

---

## 🛠️ Tech Stack

- **Backend:** FastAPI (Python 3.10+), async httpx, OpenRouter API
- **Frontend:** React + Vite, react‑markdown for rendering
- **Storage:** JSON files in `data/conversations/`
- **Package Management:** uv for Python, npm for JavaScript

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. Please ensure that any new code follows the existing style and includes appropriate documentation.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
