from langchain_community.document_loaders import WebBaseLoader


def load_website(url: str):
    """
    Loads website content and returns documents.
    """

    loader = WebBaseLoader(url)

    documents = loader.load()

    return documents