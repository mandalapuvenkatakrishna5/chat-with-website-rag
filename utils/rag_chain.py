from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


def create_rag_chain(vector_store):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )


    prompt_template = """
    Answer the question using only the given context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """


    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k":3}
        ),
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt
        }
    )


    return qa_chain