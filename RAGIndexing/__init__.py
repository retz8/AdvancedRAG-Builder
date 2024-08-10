# import different functions from other files in the package
# RAGIndexing/__init__.py
from RAGIndexing.pdf_manager import PDFManager
from RAGDatabase import RAGDatabase


class RAGIndexing:
    """
    LLM RAG Indexing class

    @args:
    - file_type: dataset type (pdf, csv, txt)
    - file_path
    - database_type: vector DB or knowledge graph (vector, graph)
    - config: dict (optional)

    @desc:
    - load documents
    - process documents
    - chunk documents
    - add chunks to database
    """

    def __init__(self, file_type: str, file_path: str, database_type: str, config=None):
        self.file_type = file_type
        self.file_path = file_path
        self.database_type = database_type

        # database configuration
        self.db_manager = RAGDatabase(database_type=database_type)

        self.config = config

    def test(self):
        """testing overall flow"""
        if self.file_type == "pdf":
            # initialize document manager
            pdf_manager = PDFManager(self.file_path)
            # process pdf
            docs = pdf_manager.process_pdf()

            # test database manager with vector DB and sample docs

            sample_db_manager = RAGDatabase(database_type="vector")
            test_collection_name = "test_collection"
            sample_db_manager.init_db(collection_name=test_collection_name)
            sample_db_manager.add_docs_to_db(
                docs=docs, collection_name=test_collection_name
            )

            # sample_query = "What are the limitations of CBT?"

            # query the vector database to retrieve contexts
            # self.db_manager.retrieve_contexts(
            #     query=sample_query, collection_name="test_collection"
            # )
            # print("\nSample query:", sample_query)
            # # generate response using the retrieved contexts and query
            # rag_chain = self.db_manager.generate_response(
            #     query=sample_query, collection_name="test_collection"
            # )

            # # generate response with tools
            # agent_executor = self.db_manager.generate_response_with_tools(
            #     rag_chain=rag_chain,
            #     query=sample_query,
            #     collection_name="test_collection",
            # )

            # delete collection
            # self.db_manager.delete_db(collection_name="test_collection")

    # 24.08.06 참고 URL
    # Neo4j로 VectorStore를 구현하는 방법
    # https://python.langchain.com/v0.2/docs/integrations/vectorstores/neo4jvector/#working-with-vectorstore
