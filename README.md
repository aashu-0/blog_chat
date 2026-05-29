# RAG blog chat

Chat with any blogpost using retrieval-augmented generation(RAG). paste a url, load it, ask questions.

---

## stack

- **FastAPI** — serves the api and static frontend
- **Cohere** — text embeddings (`embed-english-v3.0`)
- **Pinecone** — vector store; each session gets its own namespace
- **Gemini 2.5 flash lite** — answer generation
- **LangChain** — wires retrieval + llm together

---

## how it works

1. `POST /load` — fetches the url, extracts visible text, splits into ~1000-char chunks, embeds and upserts into pinecone under a unique namespace. returns a `session_id`.
2. `POST /chat` — embeds the question, retrieves top-4 chunks from that session's namespace, builds a prompt with conversation history, generates a grounded answer from gemini.
3. `DELETE /session/{id}` — wipes all vectors for that namespace from pinecone and removes the session from memory.

---

## files

| file | what it does |
|---|---|
| `rag.py` | all rag logic — loading, splitting, embedding, retrieval, generation |
| `main.py` | fastapi app, route definitions, request/response models |
| `pyproject.toml` | project metadata and dependencies (managed by uv) |
| `static/index.html` | single-page chat ui |

---

## setup

### 1. install uv

```bash
# macOS / linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. clone and install

```bash
git clone https://github.com/aashu-0/blog_chat.git
cd blog-chat

uv sync
```

### 3. env vars

create a `.env` file in the project root:

```
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
COHERE_API_KEY=
GOOGLE_API_KEY=
```

### 4. run

```bash
uv run uvicorn main:app --reload
```

open `http://localhost:8000`

---

## modifying the project

### swapping the embedding model

in `rag.py`, replace the `CohereEmbeddings` block. pick any langchain-compatible embeddings class (openai, huggingface, etc.), add the package with `uv add`, update the relevant env var.

```python
# example: switch to openai
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

the pinecone index dimension must match the new model's output size — create a new index if it doesn't.

### swapping the LLM

in `rag.py`, replace the `ChatGoogleGenerativeAI` block with any langchain chat model:

```python
# example: switch to openai
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

add the package with `uv add langchain-openai`.

### changing chunk size

in `rag.py`, edit the splitter config in `_split()`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # characters per chunk
    chunk_overlap=200, # overlap between consecutive chunks
)
```

larger chunks = more context per retrieval but noisier. smaller = more precise but may miss surrounding context.

### changing how many chunks are retrieved

in `chat()` in `rag.py`:

```python
retrieved_docs = vector_store.similarity_search(question, k=4)  # change k
```

---

## notes

- sessions are in-memory only — restarting the server loses session mappings (vectors stay in pinecone until explicitly deleted)
- conversation history is capped at last 10 turns to keep context window sane
- the llm is instructed to answer only from retrieved context, not general knowledge
