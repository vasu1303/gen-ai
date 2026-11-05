
import streamlit as st

from src.rag_pipeline import (
    initialize_rag,
    answer_question,
    save_feedback,
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Multimodal RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multimodal RAG Chatbot")

st.caption(
    "Ask questions about research papers, "
    "Transformer architecture, diagrams and figures."
)


# --------------------------------------------------
# Initialize RAG Pipeline
# --------------------------------------------------

@st.cache_resource
def load_rag_resources():
    """
    Load embedding model and both FAISS vector stores.
    Resources remain cached across Streamlit reruns.
    """
    return initialize_rag()


with st.spinner("Loading RAG models and vector stores..."):

    (
        embedding_model,
        index,
        metadata,
        image_index,
        image_metadata,
    ) = load_rag_resources()


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Feedback Display Function
# --------------------------------------------------

def display_feedback(message):

    question = message.get("question")

    if not question:
        return

    # If feedback has already been submitted
    if message.get("feedback"):

        st.caption(
            f"Your feedback: {message['feedback']}"
        )

        return

    st.write("Was this answer helpful?")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "👍 Helpful",
            key=f"helpful_{message['id']}",
        ):

            saved = save_feedback(
                feedback="Helpful",
                question=question,
            )

            if saved:

                message["feedback"] = "Helpful"

                st.success("Thank you for your feedback!")

                st.rerun()

            else:

                st.warning(
                    "Could not save feedback. "
                    "Please try again."
                )

    with col2:

        if st.button(
            "👎 Not Helpful",
            key=f"not_helpful_{message['id']}",
        ):

            saved = save_feedback(
                feedback="Not Helpful",
                question=question,
            )

            if saved:

                message["feedback"] = "Not Helpful"

                st.success("Thank you for your feedback!")

                st.rerun()

            else:

                st.warning(
                    "Could not save feedback. "
                    "Please try again."
                )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("About the Project")

    st.write(
        "A multimodal Retrieval-Augmented Generation "
        "system using research papers."
    )

    st.divider()

    st.subheader("Models")

    st.write("Text Model: Qwen2.5 1.5B")
    st.write("Vision Model: Qwen3-VL 4B")
    st.write("Embedding Model: BGE-small-en-v1.5")

    st.divider()

    st.subheader("Vector Databases")

    st.write(f"Text vectors: {index.ntotal}")
    st.write(f"Image vectors: {image_index.ntotal}")

    st.divider()

    if st.button(
        "Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# Display Previous Messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        # Display message text
        st.markdown(message["content"])

        # Display retrieved image if available
        if message.get("image_path"):

            st.image(
                message["image_path"],
                caption=message.get(
                    "image_filename",
                    "Retrieved image"
                ),
                width="stretch",
            )

        # Display sources and feedback for assistant
        if message["role"] == "assistant":

            # Display sources
            if message.get("sources"):

                with st.expander("View Sources"):

                    for source in message["sources"]:

                        st.write(
                            f"PDF: {source['source']}"
                        )

                        st.write(
                            f"Page: {source['page']}"
                        )

                        st.write(
                            f"Similarity: {source['score']:.4f}"
                        )

                        st.divider()

            # Display performance
            if message.get("performance"):

                st.caption(
                    f"Retrieval: "
                    f"{message['performance']['retrieval']:.2f}s "
                    f"| Generation: "
                    f"{message['performance']['generation']:.2f}s"
                )

            # Display feedback buttons
            display_feedback(message)


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your research papers..."
)


if question:

    # Display user question
    st.session_state.messages.append({

        "role": "user",

        "content": question,

    })

    with st.chat_message("user"):

        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching research papers and generating an answer..."
        ):

            try:

                result = answer_question(
                    question=question,
                    embedding_model=embedding_model,
                    index=index,
                    metadata=metadata,
                    image_index=image_index,
                    image_metadata=image_metadata,
                )

                answer = result["answer"]

                # Get retrieved image information
                image_path = result.get("image_path")

                image_filename = result.get("image_filename")

                # Display retrieved image
                if image_path:

                    st.subheader("Retrieved Image")

                    st.image(
                        image_path,
                        caption=image_filename or "Retrieved image",
                        width="stretch",
                    )

                # Display generated answer
                st.markdown(answer)

                # Display sources
                if result.get("sources"):

                    with st.expander("View Sources"):

                        for source in result["sources"]:

                            st.write(
                                f"PDF: {source['source']}"
                            )

                            st.write(
                                f"Page: {source['page']}"
                            )

                            st.write(
                                f"Similarity: {source['score']:.4f}"
                            )

                            st.divider()

                # Display performance
                st.caption(
                    f"Retrieval: {result['retrieval_time']:.2f}s "
                    f"| Generation: {result['llm_time']:.2f}s"
                )

                # Create assistant message
                assistant_message = {

                    "id": len(st.session_state.messages),

                    "role": "assistant",

                    "content": answer,

                    "question": question,

                    "feedback": None,

                    "image_path": image_path,

                    "image_filename": image_filename,

                    "sources": result.get("sources", []),

                    "performance": {

                        "retrieval": result["retrieval_time"],

                        "generation": result["llm_time"],

                    },

                }

                # Save assistant response in session state
                st.session_state.messages.append(
                    assistant_message
                )

                # Display feedback buttons immediately
                display_feedback(assistant_message)

            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )
