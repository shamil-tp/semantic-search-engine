# Semantic Search Engine

A modular, reproducible semantic search engine in Python spanning from raw document ingestion and preprocessing to lexical (TF-IDF), dense vector embeddings (SentenceTransformers), and reciprocal rank fusion hybrid retrieval with a responsive Flask web interface, ready for Ubuntu VPS deployment.

---

## 1. Architecture Overview

```text
├── src/
│   ├── ingestion/       # Loader, cleaner, overlapping chunker, schema, profiling
│   ├── query/           # Intent classifier (SVM), NER, spelling & controlled expansion
│   ├── retrieval/
│   │   ├── lexical.py   # TF-IDF sparse matrix cosine similarity
│   │   ├── dense.py     # SentenceTransformers bi-encoder exact search
│   │   ├── ann.py       # scikit-learn NearestNeighbors vector search & safe persistence
│   │   ├── hybrid.py    # Reciprocal Rank Fusion (RRF) combining lexical + dense
│   │   └── cli.py       # CLI commands for building, searching, evaluating
│   ├── evaluation/      # Recall@k, MRR, nDCG@k benchmark evaluator
│   └── api/
│       ├── app.py       # Flask application factory, /health, /api/search, /
│       ├── templates/   # Jinja2 HTML web interface (index.html)
│       └── static/      # Vanilla CSS (style.css) and JavaScript (app.js)
├── deploy/
│   ├── semantic-search.service  # systemd service unit for Gunicorn
│   └── nginx.conf.example       # Nginx reverse proxy configuration
├── data/sample_corpus/          # Demonstration corpus (.txt, .md)
├── pyproject.toml               # Python 3.12+ project configuration & dependencies
└── tests/                       # 50 automated pytest test cases
```

---

## 2. Local Setup (Windows Development)

### Prerequisites
- Python 3.10+ (tested on Python 3.12 and 3.14)
- Git

### Installation
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install package in editable mode with development tools
pip install -e ".[dev]"
```

### Running Tests
```powershell
pytest -v
```

### Running the Web Application
```powershell
# Launch local development server
python -m src.api.app
# Open in browser: http://127.0.0.1:8000
```

---

## 3. Data Ingestion & Index Creation Workflow

### Step 1: Ingest & Chunk Documents

**From a folder of text/markdown files:**
```powershell
python -m src.ingestion.cli `
  --input-dir data/sample_corpus `
  --output-dir data/processed `
  --chunk-size 500 `
  --chunk-overlap 50
```

**Or directly from a web crawler export (`.json` or `.jsonl`):**
```powershell
python -m src.ingestion.cli `
  --crawler-file data/crawler.sites.json `
  --output-dir data/processed `
  --limit 5000 # (optional limit)
```

### Step 2: Build Lexical TF-IDF Index
```powershell
python -m src.retrieval.cli build `
  --chunks-file data/processed/chunks.jsonl `
  --index-dir artifacts/indices/lexical_sample
```

### Step 3: Build Dense / Vector Index
```powershell
python -m src.retrieval.cli build-dense `
  --chunks-file data/processed/chunks.jsonl `
  --index-dir artifacts/indices/dense_sample
```

---

## 4. Web Interface & API Endpoints

### Web Interface
Navigate to `http://127.0.0.1:8000` to access the search interface. Features:
- Keyboard support (Enter to search, accessible labels)
- Real-time loading indicators
- Score ranking and metadata breakdown (file name, chunk index, document ID)
- Optional extension filters (`.md`, `.txt`)

### REST API

#### 1. Health Check
`GET /health`
```json
{
  "search_mode": "lexical",
  "status": "ok"
}
```

#### 2. Search Endpoint
`POST /api/search` (or `POST /api/v1/search`)
```json
{
  "query": "vector embeddings in semantic search",
  "top_k": 5,
  "filters": {
    "extension": ".md"
  }
}
```
**Response Schema**:
```json
{
  "count": 2,
  "latency_ms": 1.25,
  "query": "vector embeddings in semantic search",
  "top_k": 5,
  "results": [
    {
      "rank": 1,
      "chunk_id": "a3090451a0a256d7_c1",
      "document_id": "a3090451a0a256d7",
      "score": 0.3142,
      "text": "using for search. ## Key Principles 1. Contextual Understanding...",
      "source_path": ".../doc1_semantic_search.md",
      "chunk_index": 1,
      "metadata": {
        "file_name": "doc1_semantic_search.md",
        "extension": ".md"
      }
    }
  ]
}
```

---

## 5. Production Deployment (Ubuntu VPS)

### Prerequisites on Ubuntu VPS
```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3-pip nginx certbot python3-certbot-nginx
```

### Step 1: Clone Repository & Create User
```bash
# Setup unprivileged service user
sudo useradd -r -s /bin/false -d /var/www/semantic-search-engine www-data 2>/dev/null || true

# Place code
sudo git clone <YOUR_REPO_URL> /var/www/semantic-search-engine
cd /var/www/semantic-search-engine

# Setup Python 3.12 virtual environment & CPU torch
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[production]"
```

### Step 2: Configure Environment & Generate Indexes
```bash
cp .env.example .env
# Edit .env for production settings
nano .env

# Generate production indexes prior to launching the service
python -m src.ingestion.cli -i data/sample_corpus -o data/processed
python -m src.retrieval.cli build -c data/processed/chunks.jsonl -o artifacts/indices/lexical_sample
python -m src.retrieval.cli build-dense -c data/processed/chunks.jsonl -o artifacts/indices/dense_sample

# Create log directory and set permissions
sudo mkdir -p /var/log/semantic-search
sudo chown -R www-data:www-data /var/www/semantic-search-engine /var/log/semantic-search
```

### Step 3: Setup systemd Service
```bash
sudo cp deploy/semantic-search.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now semantic-search
sudo systemctl status semantic-search
```

### Step 4: Setup Nginx & HTTPS
```bash
sudo cp deploy/nginx.conf.example /etc/nginx/sites-available/semantic-search
# Replace server_name in /etc/nginx/sites-available/semantic-search with your actual domain
sudo ln -s /etc/nginx/sites-available/semantic-search /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Enable SSL/HTTPS with Let's Encrypt
sudo certbot --nginx -d search.yourdomain.com
```

### Memory & Gunicorn Sizing
In `deploy/semantic-search.service`, Gunicorn is configured with `--workers 2 --threads 4 --worker-class gthread`.
> **VPS Memory Note**: Because dense embedding models (e.g., SentenceTransformers) load neural network weights and PyTorch runtime into RAM, each process worker requires ~300–500MB of resident memory. Using threads within 2 workers allows concurrent request handling without duplicating model weights across many processes.

---

## 6. Persistence & Backup Guidance

- Persisted indexes reside in `artifacts/indices/` (e.g. `lexical_sample/` and `dense_sample/`).
- Embeddings are saved safely as binary NumPy arrays (`embeddings.npy`) accompanied by chunk catalogs (`chunks.jsonl`) and metadata manifests (`vector_metadata.json`), avoiding pickle vulnerabilities.
- To backup the search database:
  ```bash
  tar -czvf semantic_search_backup_$(date +%F).tar.gz artifacts/indices/ data/processed/
  ```
