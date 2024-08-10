from .config import Config
from .database import VectorDBManager, KnowledgeGraphManager
from .file_processor import FileProcessor


class AdvancedRAG:
    """
    AdvancedRAG class for advanced LLM RAG model
    """

    def __init__(self, config):
        self.config = Config(config)

        # debug, check datasets
        print(f"Datasets found in configuration: '{self.config.get_dataset_names()}'")

        # initialize database managers
        self.vector_db_manager = VectorDBManager()
        self.kg_manager = KnowledgeGraphManager()

    # Indexing ---------------------------------------------------------------
    def create_indexing(self, dataset_name):
        """
        Create indexing for dataset

        @args:
        - dataset_name: dataset name
        """
        database_type = self.config.get_dataset_type(dataset_name)
        print(
            f"==== Creating Indexing '{dataset_name}' to '{database_type}' Database ===="
        )

        # if dataset is vector type,
        # create collection in ChromaDB
        if database_type == "vector":
            embedding_model = self.config.get_dataset_embedding_model(dataset_name)
            raw_files = self.config.get_dataset_files(dataset_name)

            self.vector_db_manager.init_collection(
                collection_name=dataset_name, embedding_model=embedding_model
            )

            # process files
            file_processor = FileProcessor(raw_files)
            docs = file_processor.process_files()

            # add documents to database
            self.vector_db_manager.add_docs_to_db(
                docs=docs, collection_name=dataset_name, embedding_model=embedding_model
            )

        elif database_type == "graph":
            pass
        else:
            print("Database type not supported.")
            exit(1)
