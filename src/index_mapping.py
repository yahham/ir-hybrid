mappings = {
    "properties": {
        # BEIR document ID
        "doc_id": {
            "type": "keyword"
        },
        # Title — used for BM25 lexical matching
        "title": {
            "type": "text",
            "analyzer": "english"   # english analyzer does stemming + stopword removal
        },
        # Abstract text — used for BM25 lexical matching
        "text": {
            "type": "text",
            "analyzer": "english"
        },
        # Combined title + text — what we encode into the vector
        "full_text": {
            "type": "text",
            "analyzer": "english"
        },
        # 768-dim BERT embedding of full_text — used for semantic KNN search
        "embedding": {
            "type":       "dense_vector",
            "dims":       768,
            "index":      True,
            "similarity": "cosine"
        }
    }
}
