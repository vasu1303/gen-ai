
from sentence_transformers import SentenceTransformer

# BGE-small-en-v1.5 is selected for semantic retrieval in our RAG
# application. It produces 384-dimensional embeddings and runs locally.
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_embedding_model():
    """Load the pretrained embedding model."""
    model = SentenceTransformer(MODEL_NAME)
    return model

