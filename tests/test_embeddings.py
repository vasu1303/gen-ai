
from src.embeddings import load_embedding_model

# Step 1: Load the selected embedding model
model = load_embedding_model()

# Step 2: Create a sample sentence
sample_text = "Transformers use attention mechanisms to process text."

# Step 3: Convert the sentence into an embedding
embedding = model.encode(sample_text)

# Step 4: Display the results
print("Model loaded successfully!")
print("Embedding shape:", embedding.shape)
print("Number of dimensions:", len(embedding))
print("First 10 values:", embedding[:10])
