import os

def apply_mac_fixes():
    # Fix for HuggingFace Tokenizers deadlocks in multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Fix for PyTorch/macOS segfaults when loading models (especially SentenceTransformers)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
