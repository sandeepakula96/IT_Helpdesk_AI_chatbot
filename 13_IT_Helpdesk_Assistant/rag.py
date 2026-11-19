"""
RAG over the IT support documentation.

- Splits the IT support document (04_claude_anthropic/Docs/Day-12/it_sector.txt)
  into sections (one "## " heading = one chunk).
- Embeds the chunks with Voyage AI and stores them in a ChromaDB vector database
  (persisted in vector_db/ next to this file).
- Re-embeds automatically when any document changes.
- search() embeds the question and finds the closest chunks (optionally reranked).
"""
import hashlib
import time
from pathlib import Path

import chromadb
import voyageai

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "Docs" / "Day-12"
SOURCE_FILES = [DATA_DIR / "it_sector.txt"]  # documents used for RAG
DB_DIR = BASE_DIR / "vector_db"
COLLECTION = "it_support_docs"

EMBED_MODEL = "voyage-4-lite"
RERANK_MODEL = "rerank-2"
# Reranking costs one extra Voyage request per search. Voyage's free tier
# (no payment method added) allows only 3 requests per minute, so it is off by
# default; with a handful of short sections, vector search alone ranks well.
USE_RERANK = False


def load_sections():
    """Split each source document into sections at '## ' headings."""
    sections = []
    for path in SOURCE_FILES:
        if not path.exists():
            raise RuntimeError(f"Knowledge base file not found: {path}")
        current_title, lines = "Introduction", []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if any(l.strip() for l in lines):
                    sections.append((path.name, current_title, "\n".join(lines).strip()))
                current_title, lines = line[3:].strip(), [line]
            elif not line.startswith("# "):  # skip the document's main title
                lines.append(line)
        if any(l.strip() for l in lines):
            sections.append((path.name, current_title, "\n".join(lines).strip()))
    return sections


class KnowledgeBase:
    def __init__(self, voyage_api_key):
        self.vo = voyageai.Client(api_key=voyage_api_key, max_retries=2)
        self._query_cache = {}
        self.db = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self._build_or_load()

    def _build_or_load(self):
        sections = load_sections()
        if not sections:
            raise RuntimeError("The knowledge base documents are empty.")
        fingerprint = hashlib.sha256(
            "\x00".join(f"{s}|{t}|{x}" for s, t, x in sections).encode("utf-8")
        ).hexdigest()

        try:
            existing = self.db.get_collection(COLLECTION)
            if (existing.metadata or {}).get("fingerprint") == fingerprint:
                return existing  # documents unchanged: reuse stored embeddings
            self.db.delete_collection(COLLECTION)
        except Exception:
            pass  # collection does not exist yet

        collection = self.db.create_collection(
            COLLECTION,
            metadata={"fingerprint": fingerprint},
            configuration={"hnsw": {"space": "cosine"}},
        )
        texts = [text for _, _, text in sections]
        embeddings = []
        for i in range(0, len(texts), 128):
            embeddings += self.vo.embed(texts[i:i + 128], model=EMBED_MODEL,
                                        input_type="document").embeddings
        collection.add(
            ids=[f"chunk-{i}" for i in range(len(texts))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": s, "section": t} for s, t, _ in sections],
        )
        return collection

    def count(self):
        return self.collection.count()

    def _embed_query(self, query, attempts=3, wait_seconds=20):
        key = query.strip().lower()
        if key in self._query_cache:
            return self._query_cache[key]
        for attempt in range(attempts):
            try:
                vec = self.vo.embed([query], model=EMBED_MODEL, input_type="query").embeddings[0]
                break
            except voyageai.error.RateLimitError:
                # Free tier: 3 requests per minute. Wait for the window to free up.
                if attempt == attempts - 1:
                    raise
                time.sleep(wait_seconds)
        self._query_cache[key] = vec
        return vec

    def search(self, query, k_search=8, k_final=3):
        """Return the best matching sections as [{source, section, text, score}]."""
        res = self.collection.query(query_embeddings=[self._embed_query(query)],
                                    n_results=min(k_search, self.collection.count()))
        docs, metas = res["documents"][0], res["metadatas"][0]
        dists = res["distances"][0]
        if not docs:
            return []
        if not USE_RERANK:
            # Chroma already returns the closest sections first (cosine distance)
            return [{"source": m["source"], "section": m["section"], "text": d,
                     "score": round(1 - dist, 3)}
                    for d, m, dist in list(zip(docs, metas, dists))[:k_final]]
        reranked = self.vo.rerank(query, docs, model=RERANK_MODEL,
                                  top_k=min(k_final, len(docs)))
        return [
            {"source": metas[r.index]["source"], "section": metas[r.index]["section"],
             "text": docs[r.index], "score": round(r.relevance_score, 3)}
            for r in reranked.results
        ]
