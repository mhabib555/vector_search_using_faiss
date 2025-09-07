# Vector Search with FAISS

A simple semantic text search tool using FAISS and Sentence Transformers to find similar terms (e.g., vehicles, names) based on user queries.

## Features
- Encodes terms using `all-MiniLM-L6-v2` model
- Uses FAISS for fast cosine similarity search
- Saves/loads embeddings and index to/from disk
- Interactive query interface (top 5 matches)

## Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv)

## Setup
1. Clone the repo (if applicable):
   ```bash
   git clone <repository-url>
   cd vector-search-using-faiss
   ```
2. Install dependencies with uv:
   ```bash
   uv sync
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

## Usage
Run the script:
```bash
uv run main.py
```
- First run: Generates and saves embeddings/index.
- Subsequent runs: Loads saved data.
- Enter a query (e.g., "car") or "exit" to quit.

**Example**:
```plaintext
Query: car
Closest matches: ['car', 'vehicle', 'automobile', 'truck', 'van']
Distances (L2): [0.0, 0.231, 0.245, 0.312, 0.356]
```

## Project Structure
```
vector-search-using-faiss/
├── data/                # Stores embeddings.pkl and faiss_index.bin
├── src/                 # Source code
│   └── main.py          # Main script for vector search
├── pyproject.toml       # Dependency configuration
└── README.md            # Project documentation
```

## Notes
- Edit `texts` in `main.py` to customize terms.
- L2 index with normalized embeddings mimics cosine similarity.

## License
Unlicensed. Free to use and modify.