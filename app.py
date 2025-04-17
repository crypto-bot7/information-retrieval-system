import streamlit as st
import time

from streamlit import session_state

from src.helper import (get_pdf_text, get_text_chunks,
                 get_vector_store, get_conversational_chain)


def user_input(user_question):
    """
    Handle user input and display the answer.
    """
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chatHistory = response["chat_history"]
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write(f"**User:** {message.content}")
        else:
            st.write(f"**Reply:** {message.content}")

    

def main():
    st.set_page_config("Information Retrieval")
    st.header("Information-Retrieval-System 🤳")
    user_question = st.text_input("Ask a question about the uploaded documents")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF documents and CLick on the submit & process button", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                conversational_chain = get_conversational_chain(vector_store)
                st.session_state.conversation = conversational_chain


                st.success("Done!")
                st.balloons()


if __name__ == "__main__":
    main()


