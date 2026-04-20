import os
import logging
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INDEX_NAME = "beir_scifact_hybrid"

# ── Elasticsearch connection ───────────────────────────────────────────────────

es = Elasticsearch(
    os.getenv("ES_HOST"),
    basic_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD")),
    ca_certs=os.getenv("ES_CA_CERT"),
    request_timeout=60,
    max_retries=3,
    retry_on_timeout=True,
)
if not es.ping():
    raise RuntimeError("Cannot connect to Elasticsearch.")

# ── Model (cached at module level so it is loaded only once) ──────────────────

_model = None

def get_model():
    global _model
    if _model is None:
        logger.info("Loading BERT model...")
        _model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        logger.info("Model loaded.")
    return _model

# ── BM25 lexical search ────────────────────────────────────────────────────────
#
# Elasticsearch's default BM25 scoring runs over the 'title' and 'text' fields.
# The english analyzer performs stemming and stopword removal before indexing,
# so "running" and "run" match the same term — same behaviour as our Rust BM25.

def bm25_search(query: str, top_k: int) -> list:
    resp = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "multi_match": {
                    "query":  query,
                    "fields": ["title^1.5", "text"],  # title gets a small boost
                    "type":   "best_fields",           # uses the best-matching field score
                }
            },
            "size": top_k,
        },
        source_includes=["doc_id", "title", "text"],
    )
    hits = resp["hits"]["hits"]

    # Normalize BM25 scores to [0, 1] by dividing by the max score in the batch.
    # This makes scores comparable with the normalized cosine scores from BERT.
    max_score = max((h["_score"] for h in hits), default=1.0)
    for hit in hits:
        hit["_normalized_score"] = hit["_score"] / max_score if max_score > 0 else 0.0

    return hits


# ── BERT semantic search ───────────────────────────────────────────────────────
#
# Encodes the query with the same model used at index time, then performs
# approximate KNN over the stored 768-dim cosine embeddings.

def bert_search(query: str, top_k: int) -> list:
    model        = get_model()
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    resp = es.search(
        index=INDEX_NAME,
        knn={
            "field":          "embedding",
            "query_vector":   query_vector,
            "k":              top_k,
            "num_candidates": top_k * 5,
        },
        size=top_k,
        source_includes=["doc_id", "title", "text"],
    )
    hits = resp["hits"]["hits"]

    # Cosine similarity from Elasticsearch is already in [0, 1] when vectors
    # are unit-normalised, but we normalise by max anyway to be safe.
    max_score = max((h["_score"] for h in hits), default=1.0)
    for hit in hits:
        hit["_normalized_score"] = hit["_score"] / max_score if max_score > 0 else 0.0

    return hits


# ── Reciprocal Rank Fusion (RRF) ──────────────────────────────────────────────
#
# RRF merges two ranked lists without requiring score calibration.
# Each document gets a score of  Σ  1 / (k + rank)  across all lists it appears in.
#
# k = 60 is the standard value from the original RRF paper (Cormack et al., 2009).
# Higher k reduces the impact of rank position.
#
# Why RRF instead of score combination?
#   BM25 scores and cosine scores live on different scales. Normalising helps,
#   but RRF is more robust because it only cares about rank order, not magnitude.

def reciprocal_rank_fusion(
    bm25_hits:  list,
    bert_hits:  list,
    k:          int = 60,
) -> list[dict]:
    """
    Returns a list of result dicts sorted by descending RRF score.
    Each dict contains: doc_id, title, text, bm25_score, bert_score, rrf_score.
    """
    rrf_scores: dict[str, dict] = {}

    # Process BM25 hits
    for rank, hit in enumerate(bm25_hits, start=1):
        doc_id = hit["_id"]
        src    = hit.get("_source", {})
        rrf_contribution = 1.0 / (k + rank)

        if doc_id in rrf_scores:
            rrf_scores[doc_id]["rrf_score"]  += rrf_contribution
            rrf_scores[doc_id]["bm25_score"]  = hit["_normalized_score"]
        else:
            rrf_scores[doc_id] = {
                "doc_id":     doc_id,
                "title":      src.get("title", ""),
                "text":       src.get("text",  ""),
                "bm25_score": hit["_normalized_score"],
                "bert_score": 0.0,
                "rrf_score":  rrf_contribution,
            }

    # Process BERT hits
    for rank, hit in enumerate(bert_hits, start=1):
        doc_id = hit["_id"]
        src    = hit.get("_source", {})
        rrf_contribution = 1.0 / (k + rank)

        if doc_id in rrf_scores:
            rrf_scores[doc_id]["rrf_score"]  += rrf_contribution
            rrf_scores[doc_id]["bert_score"]  = hit["_normalized_score"]
        else:
            rrf_scores[doc_id] = {
                "doc_id":     doc_id,
                "title":      src.get("title", ""),
                "text":       src.get("text",  ""),
                "bm25_score": 0.0,
                "bert_score": hit["_normalized_score"],
                "rrf_score":  rrf_contribution,
            }

    return sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)


# ── Public hybrid search entry point ─────────────────────────────────────────

def hybrid_search(query: str, top_k: int = 10, rrf_k: int = 60) -> list[dict]:
    """
    Run BM25 and BERT in parallel, fuse with RRF, return top_k results.

    Parameters
    ----------
    query  : natural-language query string
    top_k  : how many results to retrieve from each sub-system before fusion
    rrf_k  : rank bias parameter for RRF (default 60)
    """
    bm25_hits = bm25_search(query, top_k=top_k)
    bert_hits = bert_search(query, top_k=top_k)
    results   = reciprocal_rank_fusion(bm25_hits, bert_hits, k=rrf_k)
    return results[:top_k]
