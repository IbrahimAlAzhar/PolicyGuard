"""
Central configuration for med_limagents_v2. Every path can be overridden with an
environment variable, so nothing has to be edited when the project moves.
"""
import os

PROJECT_DIR = os.environ.get(
    "MEDLIM_PROJECT_DIR", "/lstr/sahara/datalab-ml/ibrahim/limagents_update/medlimagents/med_limagents_v2")

# data tree: cdc_places cdc_svi census_acs cms_hospital_compare hrsa_shortage_areas pubmed pubmed_post_processed synthea unified
DATA_ROOT = os.environ.get(
    "MEDLIM_DATA_ROOT", "/lstr/sahara/datalab-ml/ibrahim/limagents_update/medlimagents/med_limagents_v2/data_full/data/data_medlim")

DEFAULT_INPUT_CSV = os.environ.get(
    "MEDLIM_INPUT_CSV", os.path.join(DATA_ROOT, "pubmed_post_processed/test_150_with_rag.csv"))
OUTPUT_DIR = os.environ.get("MEDLIM_OUTPUT_DIR", os.path.join(PROJECT_DIR, "outputs"))
CACHE_DIR = os.environ.get("MEDLIM_CACHE_DIR", os.path.join(PROJECT_DIR, "cache"))

MODELS_DIR = os.environ.get("MEDLIM_MODELS_DIR", "/lstr/sahara/datalab-ml/ibrahim/models")

# short name -> local checkpoint folder. served name == short name.
OPEN_MODELS = {
    "qwen": os.path.join(MODELS_DIR, "qwen3_8b"),
    "gemma": os.path.join(MODELS_DIR, "gemma3_27b_it"),
    "mistral": os.path.join(MODELS_DIR, "mistral_small_3_1_24b_instruct"),
}

# ---------------------------------------------------------------- input columns
COL_ID = "pmid"
COL_TITLE = "title"
COL_TEXT = "full_text_excl_lim"            # manuscript WITHOUT the author limitation section
COL_ABSTRACT = "abstract"
COL_AUTHOR_LIM = "limitations_section"     # author-stated limitations (ground truth)
COL_RAG = "rag_top3_json"                  # precomputed hybrid FAISS+BM25 top-3 (v1 precompute_rag.py)

# ---------------------------------------------------------------- public-health data
UNIFIED_DIR = os.path.join(DATA_ROOT, "unified")
PH_PATHS = {
    "hospital_risk_matrix": os.path.join(UNIFIED_DIR, "hospital_risk_matrix.parquet"),
    "tract_equity": os.path.join(UNIFIED_DIR, "tract_equity.parquet"),
    "county_equity": os.path.join(UNIFIED_DIR, "county_equity.parquet"),
    "cdc_places_county": os.path.join(DATA_ROOT, "cdc_places/places_county.csv"),
    "cdc_svi_dir": os.path.join(DATA_ROOT, "cdc_svi"),
    "census_acs_dir": os.path.join(DATA_ROOT, "census_acs"),
    "cms_hospital_dir": os.path.join(DATA_ROOT, "cms_hospital_compare"),
    "hrsa_dir": os.path.join(DATA_ROOT, "hrsa_shortage_areas"),
}

# ---------------------------------------------------------------- RAG corpus (only for --live_rag)
RAG_CORPUS_CSV = os.path.join(DATA_ROOT, "pubmed_post_processed/rag_corpus.csv")
RAG_CACHE_DIR = os.path.join(DATA_ROOT, "pubmed_post_processed/rag_cache")
RAG_EMBED_MODEL = os.environ.get("MEDLIM_RAG_EMBED_MODEL", "BAAI/bge-large-en-v1.5")

# ---------------------------------------------------------------- experiment grid
# condition -> (method, use_rag, use_public_health_data)
CONDITIONS = {
    "zs":            ("zero_shot", False, False),
    "zs_ph":         ("zero_shot", False, True),
    "rag":           ("zero_shot", True,  False),
    "rag_ph":        ("zero_shot", True,  True),
    "agents":        ("agents",    False, False),
    "agents_ph":     ("agents",    False, True),
    "agents_rag":    ("agents",    True,  False),
    "agents_rag_ph": ("agents",    True,  True),   # full PolicyGuard v2
}

JUDGE_MODEL = os.environ.get("MEDLIM_JUDGE_MODEL", "gpt-4o-mini")

os.makedirs(CACHE_DIR, exist_ok=True) if os.path.isdir(os.path.dirname(CACHE_DIR) or ".") else None
