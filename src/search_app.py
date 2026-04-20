import os
import streamlit as st
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from hybrid_search import hybrid_search, bm25_search, bert_search, reciprocal_rank_fusion

load_dotenv()

# ── Connect ────────────────────────────────────────────────────────────────────

try:
    es = Elasticsearch(
        os.getenv("ES_HOST"),
        basic_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD")),
        ca_certs=os.getenv("ES_CA_CERT"),
        request_timeout=60,
    )
except Exception as e:
    st.error(f"Elasticsearch connection error: {e}")
    st.stop()

# ── UI ─────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="BM25 + BERT Hybrid Search — SciFact", layout="wide")
    st.title("⚗️ BM25 + BERT Hybrid Search — SciFact")
    st.caption(
        "Combines Elasticsearch BM25 lexical scoring with "
        "`all-mpnet-base-v2` semantic embeddings, fused via Reciprocal Rank Fusion (RRF)."
    )

    query_text = st.text_input(
        "Enter your scientific query:",
        placeholder="e.g. smoking causes lung cancer"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Results per system", min_value=5, max_value=50, value=10)
    with col2:
        rrf_k = st.slider("RRF k parameter", min_value=10, max_value=120, value=60,
                          help="Higher values reduce the influence of rank position.")
    with col3:
        show_scores = st.checkbox("Show individual scores", value=True)

    if st.button("Search") and query_text.strip():
        with st.spinner("Running BM25 + BERT + RRF..."):
            bm25_hits = bm25_search(query_text.strip(), top_k=top_k)
            bert_hits = bert_search(query_text.strip(), top_k=top_k)
            results   = reciprocal_rank_fusion(bm25_hits, bert_hits, k=rrf_k)

        if not results:
            st.warning("No results found.")
            return

        st.subheader(f"Top {len(results)} fused results")

        for i, result in enumerate(results[:top_k], start=1):
            with st.container():
                st.markdown(f"**{i}. {result.get('title', 'No title')}**")
                st.write(result.get("text", ""))

                if show_scores:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("RRF Score",  f"{result['rrf_score']:.5f}")
                    c2.metric("BM25 Score", f"{result['bm25_score']:.4f}")
                    c3.metric("BERT Score", f"{result['bert_score']:.4f}")

                st.caption(f"Doc ID: `{result['doc_id']}`")
                st.divider()

if __name__ == "__main__":
    main()
