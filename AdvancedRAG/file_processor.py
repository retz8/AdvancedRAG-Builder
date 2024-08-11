import re
import ast

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Helpers ---------------------------------------------------------------------
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


# TODO: Move this to a utils file
def document_to_str(doc: Document):
    return f"page_content='{doc.page_content}' metadata={doc.metadata}"


# NOTE: 굳이 클래스로 만들어야하나...?
class FileProcessor:
    """
    Process different types of files

    @args:
    - files: list of files to process
    """

    def __init__(self, files):
        self.files = files

    def process_files(self):
        """
        Process files
        """
        print("==== Processing Files ====")

        docs = []

        for file in self.files:
            file_path = file["path"]
            file_type = file["type"]

            if file_type == "pdf":
                doc = self.process_pdf(file_path)
                docs.extend(doc)

            else:
                # TODO: Add support for other file types
                print(f"File type '{file_type}' not supported.")
                exit(1)

        return docs

    def process_pdf(self, file_path: str):
        """Load the dataset."""
        print("==== Processing PDF ====")

        # split into pages
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

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
