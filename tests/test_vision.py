from src.vision import analyze_image

image_path = "data/test_image.jpg"

question = "Explain the diagram shown in this image in simple words."

answer = analyze_image(
    image_path=image_path,
    question=question
)

print("\nQuestion:", question)
print("\nAnswer:", answer)