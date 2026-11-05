from src.llm import generate_answer

question = "What is Generative AI?"

context = """
Generative AI refers to artificial intelligence systems that can
generate new content, such as text, images, audio, and code.
"""

answer = generate_answer(question, context)

print("\nQuestion:", question)
print("\nGenerated Answer:\n", answer)
