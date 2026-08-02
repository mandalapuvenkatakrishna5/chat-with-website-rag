import streamlit as st
from dotenv import load_dotenv

from utils.loader import load_website
from utils.vector_store import create_vector_store
from utils.rag_chain import create_rag_chain


# Load environment variables
load_dotenv()


# Streamlit page configuration
st.set_page_config(
    page_title="Chat With Website",
    page_icon="🌐"
)


st.title("🌐 Chat With Website")
st.write(
    "Enter a website URL and ask questions based on its content."
)


# Session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


# URL input
url = st.text_input(
    "Enter Website URL"
)


# Load website button
if st.button("Load Website"):

    if url:

        with st.spinner("Reading website content..."):

            try:
                # Load website
                documents = load_website(url)


                # Create vector database
                vector_store = create_vector_store(documents)


                # Create RAG chain
                st.session_state.qa_chain = create_rag_chain(
                    vector_store
                )


                st.success(
                    "Website loaded successfully! Ask questions now."
                )


            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )

    else:
        st.warning(
            "Please enter a website URL"
        )



# Question input

question = st.text_input(
    "Ask a question"
)


if st.button("Get Answer"):

    if st.session_state.qa_chain:

        with st.spinner("Generating answer..."):

            response = st.session_state.qa_chain.invoke(
                {
                    "query": question
                }
            )


            st.write(
                response["result"]
            )

    else:

        st.warning(
            "Please load a website first."
        )