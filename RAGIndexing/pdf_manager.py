# RAGIndexing/pdf_manager.py

import re
import ast

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# TODO: Move this to a utils file
def preprocess_text(text):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r"\s+", " ", text)
    return text


# TODO: Move this to a utils file
def str_to_document(text: str):
    # Split the string into page_content and metadata
    page_content_part, metadata_part = text.split(" metadata=")

    # Extract page content
    page_content = page_content_part.split("page_content=", 1)[1].strip("'")

    # parse metadata string to dictionary
    metadata = ast.literal_eval(metadata_part)

    return Document(page_content=page_content, metadata=metadata)


class PDFManager:
    """
    Class for loading and processing PDF documents.

    @args:
    - file_path

    @desc:
    - load PDF file
    - split into pages
    - split pages into chunks (tokens)

    @return:
    - list of Document objects (tokens)
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        # initialize loader with PyPDFLoader
        self.loader = PyPDFLoader(file_path)

    def process_pdf(self):
        """Load the dataset."""
        print("==== Processing PDF ====")

        # split into pages
        pages = self.loader.load_and_split()

        # split pages into tokens using tiktoken encoder
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-2",
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=100,
            chunk_overlap=0,
        )

        raw_chunks = text_splitter.split_documents(pages)

        # Convert Document objects into strings
        chunks = [str(doc) for doc in raw_chunks]
        # Preprocess the text
        chunks = [preprocess_text(chunk) for chunk in chunks]
        # convert strings to Document objects
        docs = [str_to_document(chunk) for chunk in chunks]

        print("Number of splitted tokens:", len(docs))

        return docs
