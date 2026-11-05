
import os
import re
import time
from collections import deque

from src.embeddings import load_embedding_model
from src.retrieval import load_vector_store, retrieve_relevant_chunks
from src.llm import generate_answer

from src.image_retrieval import (
    build_image_vector_store,
    load_image_vector_store,
    retrieve_relevant_images,
)

from src.vision import analyze_image


# Store the last 4 conversation interactions
conversation_memory = deque(maxlen=4)

def save_feedback(feedback: str, question: str):
    """Save feedback for the corresponding conversation interaction."""

    for interaction in reversed(conversation_memory):

        if interaction["question"] == question:

            interaction["feedback"] = feedback

            print("\n--- FEEDBACK SAVED ---")
            print("Question:", question)
            print("Feedback:", feedback)
            print("----------------------\n")

            return True

    print("Feedback could not be matched to a conversation.")
    return False

def print_conversation_memory():
    """Display the current four interactions stored in memory."""

    print("\n--- CURRENT CONVERSATION MEMORY ---")

    for i, interaction in enumerate(conversation_memory, start=1):
        print(f"{i}. {interaction['question']}")

    print("Total interactions:", len(conversation_memory))
    print("-----------------------------------\n")

def is_visual_question(question: str) -> bool:
    """Check whether the user explicitly refers to a visual."""

    visual_keywords = {
        "image", "diagram", "figure", "graph",
        "chart", "visual", "illustration", "picture",
    }

    question_words = set(
        re.findall(r"\b[a-zA-Z]+\b", question.lower())
    )

    return bool(question_words.intersection(visual_keywords))


def get_previous_visual():
    """Find the most recent interaction containing an image."""

    for interaction in reversed(conversation_memory):
        if interaction.get("image_path"):
            return interaction

    return None


def is_visual_followup(question: str) -> bool:
    """
    Detect a question that likely refers to a previously
    discussed image or diagram.
    """

    previous_visual = get_previous_visual()

    if not previous_visual:
        return False

    followup_keywords = {
        "it", "this", "that", "these", "those",
        "above", "there", "shown", "second",
        "first", "third", "layer", "component",
        "block", "arrow", "connection", "output",
        "input", "attention", "encoder", "decoder",
    }

    question_words = set(
        re.findall(r"\b[a-zA-Z]+\b", question.lower())
    )

    return bool(question_words.intersection(followup_keywords))


def initialize_rag():
    """Load embedding model and text/image FAISS stores."""

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Loading text FAISS vector store...")
    index, metadata = load_vector_store()

    try:
        print("\nLoading image FAISS vector store...")
        image_index, image_metadata = load_image_vector_store()

    except FileNotFoundError:
        print("\nImage vector store not found. Creating it...")

        image_index, image_metadata = build_image_vector_store(
            embedding_model
        )

    print("\nRAG pipeline initialized successfully!")

    return (
        embedding_model,
        index,
        metadata,
        image_index,
        image_metadata,
    )


def answer_question(
    question: str,
    embedding_model,
    index,
    metadata,
    image_index,
    image_metadata,
    top_k: int = 3,
) -> dict:
    """
    Retrieve relevant text or images and generate answers.

    Text questions use Qwen2.5.
    Visual questions and visual follow-ups use Qwen3-VL.

    The chatbot remembers the last four interactions.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    question = question.strip()

    # --------------------------------------------------
    # Step 1: Prepare conversation history
    # --------------------------------------------------

    conversation_history = "\n\n".join(
        [
            (
                f"User: {interaction['question']}\n"
                f"Assistant: {interaction['answer']}"
                f"User Feedback: {interaction.get('feedback', 'No feedback provided')}"
            )
            for interaction in conversation_memory
        ]
    )

    # --------------------------------------------------
    # Step 2: Detect visual questions and follow-ups
    # --------------------------------------------------

    previous_visual = get_previous_visual()

    explicit_visual_question = is_visual_question(question)

    visual_followup = (
        not explicit_visual_question
        and is_visual_followup(question)
    )

    image_path = None
    image_filename = None
    selected_image = None

    retrieval_time = 0.0

    # --------------------------------------------------
    # Step 3: Reuse previous image for follow-up questions
    # --------------------------------------------------

    if visual_followup and previous_visual:

        print("\nVisual follow-up detected.")
        print("Reusing previously retrieved image.")

        image_path = previous_visual["image_path"]
        image_filename = previous_visual["image_filename"]

        selected_image = {
            "source": previous_visual["source"],
            "page": previous_visual["page"],
            "score": previous_visual["score"],
        }

    # --------------------------------------------------
    # Step 4: Retrieve a new image for explicit visual query
    # --------------------------------------------------

    elif explicit_visual_question:

        print("\nVisual question detected.")

        image_retrieval_start = time.time()

        relevant_images = retrieve_relevant_images(
            question=question,
            index=image_index,
            metadata=image_metadata,
            model=embedding_model,
            top_k=1,
        )

        retrieval_time = time.time() - image_retrieval_start

        print(
            f"\nImage retrieval time: {retrieval_time:.2f} seconds"
        )

        if relevant_images:

            selected_image = relevant_images[0]

            image_path = selected_image["image_path"]
            image_filename = selected_image["filename"]

        else:
            print("\nNo relevant image found. Switching to text RAG.")

    # --------------------------------------------------
    # Step 5: Analyze image using Qwen3-VL
    # --------------------------------------------------

    if image_path and selected_image:

        print(f"\nRetrieved image: {image_filename}")
        print(f"Source: {selected_image['source']}")
        print(f"Page: {selected_image['page']}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Retrieved image does not exist: {image_path}"
            )

        vision_start = time.time()

        answer = analyze_image(
            image_path=image_path,
            question=question,
            conversation_history=conversation_history,
        )

        vision_time = time.time() - vision_start

        print(
            f"\nVision model generation time: "
            f"{vision_time:.2f} seconds"
        )

        # Save visual interaction in memory
        conversation_memory.append({
            "question": question,
            "answer": answer,
            "image_path": image_path,
            "image_filename": image_filename,
            "source": selected_image["source"],
            "page": selected_image["page"],
            "score": selected_image["score"],
        })

# Display current memory
        print_conversation_memory()

        return {
            "question": question,
            "answer": answer,
            "image_path": image_path,
            "image_filename": image_filename,
            "sources": [
                {
                    "source": selected_image["source"],
                    "page": selected_image["page"],
                    "score": selected_image["score"],
                }
            ],
            "retrieval_time": retrieval_time,
            "llm_time": vision_time,
        }

    # --------------------------------------------------
    # Step 6: Text-based RAG
    # --------------------------------------------------

    print("\nProcessing text-based question.")

    if conversation_memory:

        recent_history = "\n".join(
            [
                f"User: {interaction['question']}"
                for interaction in conversation_memory
            ]
        )

        search_query = (
            f"Recent conversation:\n{recent_history}\n"
            f"Current question: {question}"
        )

    else:
        search_query = question

    # --------------------------------------------------
    # Step 7: Retrieve relevant text chunks
    # --------------------------------------------------

    retrieval_start = time.time()

    retrieved_chunks = retrieve_relevant_chunks(
        question=search_query,
        index=index,
        metadata=metadata,
        model=embedding_model,
        top_k=top_k,
    )

    retrieval_time = time.time() - retrieval_start

    print(f"\nRetrieval time: {retrieval_time:.2f} seconds")

    # --------------------------------------------------
    # Step 8: Prepare context
    # --------------------------------------------------

    context = "\n\n".join(
        [
            (
                f"Source: {chunk['source']}, "
                f"Page: {chunk['page']}\n"
                f"{chunk['text'][:1200]}"
            )
            for chunk in retrieved_chunks[:2]
        ]
    )

    # --------------------------------------------------
    # Step 9: Generate answer using Qwen2.5
    # --------------------------------------------------

    llm_start = time.time()

    answer = generate_answer(
        question=question,
        context=context,
        conversation_history=conversation_history,
    )

    llm_time = time.time() - llm_start

    print(f"LLM generation time: {llm_time:.2f} seconds")

    # --------------------------------------------------
    # Step 10: Save interaction in memory
    # --------------------------------------------------

    conversation_memory.append({
        "question": question,
        "answer": answer,
        "image_path": None,
        "image_filename": None,
        "source": None,
        "page": None,
        "score": None,
    })
    print_conversation_memory()

    # --------------------------------------------------
    # Step 11: Return response
    # --------------------------------------------------

    return {
        "question": question,
        "answer": answer,
        "image_path": None,
        "image_filename": None,
        "sources": [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "score": chunk["score"],
            }
            for chunk in retrieved_chunks[:2]
        ],
        "retrieval_time": retrieval_time,
        "llm_time": llm_time,
    }


# --------------------------------------------------
# Command Line Interface
# --------------------------------------------------

if __name__ == "__main__":

    (
        embedding_model,
        index,
        metadata,
        image_index,
        image_metadata,
    ) = initialize_rag()

    print("\nAsk questions about your research papers.")
    print("Type 'exit' to stop.")
    print("The chatbot remembers the last 4 interactions.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            result = answer_question(
                question=question,
                embedding_model=embedding_model,
                index=index,
                metadata=metadata,
                image_index=image_index,
                image_metadata=image_metadata,
            )

            print("\nAssistant:", result["answer"])

            if result.get("image_path"):
                print("Retrieved image:", result["image_path"])

            print("\nSources:")

            for source in result["sources"]:
                print(
                    f"- {source['source']}, "
                    f"page {source['page']}, "
                    f"similarity {source['score']:.4f}"
                )

            print("\nPerformance:")
            print(
                f"Retrieval: {result['retrieval_time']:.2f} seconds"
            )
            print(
                f"LLM generation: {result['llm_time']:.2f} seconds"
            )

            print()

        except Exception as error:
            print(f"\nError: {error}\n")
