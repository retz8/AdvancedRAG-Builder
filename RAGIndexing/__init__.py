# import different functions from other files in the package
# RAGIndexing/__init__.py

class RAGIndexing:
    def __init__(self, database_type='vector'):
        self.text = "Hello, World!"
        self.database_type = database_type
    
    def test(self):
        """ testing function """
        print("Hello, Worldd!")
    
    def print_text(self):
        print(self.text)

    # 24.08.06 참고 URL
    # Neo4j로 VectorStore를 구현하는 방법
    # https://python.langchain.com/v0.2/docs/integrations/vectorstores/neo4jvector/#working-with-vectorstore
    
    # def __init__(self, db_type='vector', config=None):
    #     # Initialize data loader, document processor, chunker, and database managers
    #     self.data_loader = DataLoader(config)
    #     self.document_processor = DocumentProcessor()
    #     self.chunker = Chunker()
        
    #     if db_type == 'vector':
    #         self.db_manager = VectorDBManager(config)
    #     elif db_type == 'graph':
    #         self.db_manager = KnowledgeGraphManager(config)
    #     else:
    #         raise ValueError("Unsupported database type. Choose 'vector' or 'graph'.")

    # def load_data(self, dataset_path):
    #     """Load the dataset."""
    #     return self.data_loader.load(dataset_path)

    # def process_document(self, document):
    #     """Process the document."""
    #     return self.document_processor.process(document)

    # def chunk_document(self, document):
    #     """Chunk the document into smaller parts."""
    #     return self.chunker.chunk(document)

    # def add_chunks_to_db(self, chunks):
    #     """Add chunks to the database."""
    #     self.db_manager.add_chunks(chunks)

    # def initialize_db(self):
    #     """Initialize the database."""
    #     self.db_manager.initialize()

    # def configure(self, config):
    #     """Reconfigure the RAGIndexing instance."""
    #     self.data_loader.update_config(config)
    #     self.db_manager.update_config(config)


