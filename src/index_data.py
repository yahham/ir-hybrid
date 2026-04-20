import os
import json
import logging
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from index_mapping import mappings

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Elasticsearch connection ───────────────────────────────────────────────────

es = Elasticsearch(
    os.getenv("ES_HOST"),
    basic_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD")),
    ca_certs=os.getenv("ES_CA_CERT"),
    request_timeout=120,
    max_retries=3,
    retry_on_timeout=True,
)
if not es.ping():
    raise RuntimeError("Cannot connect to Elasticsearch. Check your .env settings.")
logger.info("Connected to Elasticsearch.")

INDEX_NAME   = "beir_scifact_hybrid"
DATASET_PATH = os.getenv("BEIR_DATASET_PATH", "../../datasets/scifact")
CORPUS_FILE  = os.path.join(DATASET_PATH, "corpus.jsonl")
BATCH_SIZE   = 64

# ── Load corpus ────────────────────────────────────────────────────────────────

logger.info(f"Loading corpus from {CORPUS_FILE} ...")
corpus = []
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        doc       = json.loads(line)
        title     = doc.get("title", "") or ""
        text      = doc.get("text",  "") or ""
        full_text = f"{title} {text}".strip() if title else text
        corpus.append({
            "doc_id":    doc["_id"],
            "title":     title,
            "text":      text,
            "full_text": full_text,
        })
logger.info(f"Loaded {len(corpus)} documents.")

# ── Load BERT model ────────────────────────────────────────────────────────────
# Using the same model as the standalone BERT app so the semantic component
# is identical — the only difference between BERT and BM25+BERT is the fusion.

logger.info("Loading BERT model (all-mpnet-base-v2)...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
logger.info("Model loaded.")

# ── Create / recreate index ────────────────────────────────────────────────────

if es.indices.exists(index=INDEX_NAME):
    logger.info(f"Deleting existing index '{INDEX_NAME}'...")
    es.indices.delete(index=INDEX_NAME)

es.indices.create(index=INDEX_NAME, mappings=mappings)
logger.info(f"Index '{INDEX_NAME}' created.")

# ── Encode and bulk-index ─────────────────────────────────────────────────────
#
# This index stores BOTH full-text fields (for BM25) AND dense vectors (for
# BERT KNN). Elasticsearch handles both query types on the same index.

def generate_actions(corpus, model, batch_size):
    """
    Generator: encodes documents in batches and yields bulk-index actions.
    """
    for batch_start in range(0, len(corpus), batch_size):
        batch      = corpus[batch_start : batch_start + batch_size]
        texts      = [doc["full_text"] for doc in batch]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for doc, emb in zip(batch, embeddings):
            yield {
                "_index": INDEX_NAME,
                "_id":    doc["doc_id"],
                "_source": {
                    "doc_id":    doc["doc_id"],
                    "title":     doc["title"],
                    "text":      doc["text"],
                    "full_text": doc["full_text"],
                    "embedding": emb.tolist(),
                },
            }

logger.info("Encoding and indexing documents...")
total_indexed, errors = 0, []

for ok, info in tqdm(
    helpers.streaming_bulk(
        es,
        generate_actions(corpus, model, BATCH_SIZE),
        chunk_size=100,
        raise_on_error=False,
    ),
    total=len(corpus),
    desc="Indexing",
):
    if ok:
        total_indexed += 1
    else:
        errors.append(info)

logger.info(f"Indexed {total_indexed}/{len(corpus)} documents.")
if errors:
    logger.warning(f"{len(errors)} documents failed to index.")
    for err in errors[:5]:
        logger.warning(err)

es.indices.refresh(index=INDEX_NAME)
logger.info("Index refreshed. Ready for hybrid search and evaluation.")
