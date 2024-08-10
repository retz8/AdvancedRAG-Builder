from langchain_chroma import Chroma
import chromadb
from uuid import uuid4


class VectorDBManager:
    """
    Vector Database Manager
    """

    def __init__(self):
        # initialize chroma db (for testing, we use persistent client)
        self.client = chromadb.PersistentClient()

    # Add methods for vector database operations

    def init_collection(self, collection_name, embedding_model):
        """
        Initialize database with empty collection

        @args:
        - collection_name: collection name
        - embedding_model: embedding model function
        """
        # check if the collection already exists
        # if so, delete it first
        existing_collections = self.client.list_collections()

        if any(
            collection.name == collection_name for collection in existing_collections
        ):
            print(
                f"Collection '{collection_name}' already exists. Deleting existing collection."
            )
            self.client.delete_collection(collection_name)

        # # create a new collection with embedding function
        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )
        print(f"Created new collection '{collection_name}'")

        return

    def add_docs_to_db(self, docs, collection_name, embedding_model):
        """
        Add documents to database
        """
        # error handling: collection not found
        existing_collections = self.client.list_collections()
        if not any(
            collection.name == collection_name for collection in existing_collections
        ):
            print(
                f"Collection '{collection_name}' not found. You need to initialize the database first."
            )
            return

        # get collection object (here, we are using langchain-chroma, not bare chromadb)
        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )
        # insert documents
        # 1. obtain unique ids for each document
        uuids = [str(uuid4()) for _ in range(len(docs))]
        # 2. add documents to collection
        vector_store.add_documents(documents=docs, ids=uuids)
        print(f"Added `{len(docs)}` documents to collection '{collection_name}'")

        return

    # helpful getter functions -----------------------------------------------
    def get_collection_names(self):
        """
        Get collection names
        """
        return [collection.name for collection in self.client.list_collections()]

    # def __del__(self):
    #     self.client.reset()
