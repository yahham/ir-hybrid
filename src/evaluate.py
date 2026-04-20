import os
import json
import math
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from hybrid_search import bm25_search, bert_search, reciprocal_rank_fusion

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = os.getenv("BEIR_DATASET_PATH", "../../datasets/scifact")
QUERIES_FILE = os.path.join(DATASET_PATH, "queries.jsonl")
QRELS_FILE   = os.path.join(DATASET_PATH, "qrels", "test.tsv")
TOP_K        = 100   # retrieve this many docs per sub-system before fusion

# ── Load queries ───────────────────────────────────────────────────────────────

logger.info(f"Loading queries from {QUERIES_FILE} ...")
queries = {}
with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)
        queries[q["_id"]] = q["text"]
logger.info(f"  {len(queries)} queries loaded.")

# ── Load qrels ─────────────────────────────────────────────────────────────────

logger.info(f"Loading qrels from {QRELS_FILE} ...")
qrels = {}
with open(QRELS_FILE, "r", encoding="utf-8") as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        qid, did, score = parts[0], parts[1], int(parts[2])
        if score > 0:
            qrels.setdefault(qid, {})[did] = score
logger.info(f"  {len(qrels)} queries have relevance judgements.")

# ── Run hybrid search for every query ─────────────────────────────────────────
#
# Each query goes through:
#   1. BM25 search    → top-100 lexical hits
#   2. BERT KNN       → top-100 semantic hits
#   3. RRF fusion     → single re-ranked list
#
# We then extract just the doc IDs for metric computation.

queries_to_eval = {qid: text for qid, text in queries.items() if qid in qrels}
logger.info(f"Evaluating {len(queries_to_eval)} queries...")

ranked_results = {}   # query_id -> [doc_id, doc_id, ...]  in RRF rank order

for qid, qtext in tqdm(queries_to_eval.items(), desc="Hybrid search"):
    bm25_hits = bm25_search(qtext, top_k=TOP_K)
    bert_hits = bert_search(qtext, top_k=TOP_K)
    fused     = reciprocal_rank_fusion(bm25_hits, bert_hits, k=60)
    ranked_results[qid] = [r["doc_id"] for r in fused]

# ── Metric functions ───────────────────────────────────────────────────────────

def mrr_at_k(ranked_results, qrels, k):
    total, n = 0.0, 0
    for qid, doc_ids in ranked_results.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1
        for rank, did in enumerate(doc_ids[:k], start=1):
            if did in rel:
                total += 1.0 / rank
                break
    return total / n if n > 0 else 0.0


def ndcg_at_k(ranked_results, qrels, k):
    total, n = 0.0, 0
    for qid, doc_ids in ranked_results.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1
        dcg  = sum(rel.get(did, 0) / math.log2(i + 2)
                   for i, did in enumerate(doc_ids[:k]))
        idcg = sum(r / math.log2(i + 2)
                   for i, r in enumerate(sorted(rel.values(), reverse=True)[:k]))
        if idcg > 0:
            total += dcg / idcg
    return total / n if n > 0 else 0.0


def recall_at_k(ranked_results, qrels, k):
    total, n = 0.0, 0
    for qid, doc_ids in ranked_results.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1
        hits = sum(1 for did in doc_ids[:k] if did in rel)
        total += hits / len(rel)
    return total / n if n > 0 else 0.0


def precision_at_k(ranked_results, qrels, k):
    total, n = 0.0, 0
    for qid, doc_ids in ranked_results.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1
        top_k = doc_ids[:k]
        if not top_k:
            continue
        total += sum(1 for did in top_k if did in rel) / len(top_k)
    return total / n if n > 0 else 0.0


def f1_at_k(ranked_results, qrels, k):
    p = precision_at_k(ranked_results, qrels, k)
    r = recall_at_k(ranked_results, qrels, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── Print results table ────────────────────────────────────────────────────────

ks = [1, 5, 10, 20, 100]

print("\n" + "═" * 70)
print("  BM25 + BERT (RRF) Evaluation on BEIR SciFact")
print(f"  Corpus: 5183 docs  |  Queries evaluated: {len(ranked_results)}")
print("  Fusion: Reciprocal Rank Fusion (k=60)  |  Model: all-mpnet-base-v2")
print("═" * 70)
print(f"\n{'Metric':<15}" + "".join(f"{'@'+str(k):>10}" for k in ks))
print("-" * 65)

metrics = [
    ("MRR",       mrr_at_k),
    ("NDCG",      ndcg_at_k),
    ("Recall",    recall_at_k),
    ("Precision", precision_at_k),
    ("F1",        f1_at_k),
]

for name, fn in metrics:
    row = f"{name:<15}"
    for k in ks:
        row += f"{fn(ranked_results, qrels, k):>10.4f}"
    print(row)

print(f"\nTotal queries evaluated: {len(ranked_results)}")

# ── Save raw results ───────────────────────────────────────────────────────────
# Same [[query_id, [doc_ids...]], ...] format as tfidf_results.json,
# bm25_results.json, and bert_results.json.

output_path  = os.path.join(DATASET_PATH, "bm25_bert_results.json")
results_list = [[qid, doc_ids] for qid, doc_ids in ranked_results.items()]
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results_list, f, indent=2)
logger.info(f"Raw results saved to: {output_path}")
