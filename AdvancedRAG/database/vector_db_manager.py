from langchain_chroma import Chroma
import chromadb
from uuid import uuid4

from typing import List

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain.tools.retriever import create_retriever_tool

import ast

# Set logging for the queries
import logging

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain import hub

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# Output parser will split the LLM result into a list of queries (for multiple queries)
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


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


class VectorDBManager:
    """
    Vector Database Manager
    """

    def __init__(self):
        # initialize chroma db (for testing, we use persistent client)
        self.client = chromadb.PersistentClient()

    # Add methods for vector database operations

    # Indexing ---------------------------------------------------------------
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
            self.reset_manager()

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
        print("==== Adding documents to collection ====")
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
        print(
            f"Successfully added `{len(docs)}` documents to collection '{collection_name}'"
        )

        return

    # Retrieval ---------------------------------------------------------------
    def retrieve_contexts(self, query, collection_name, embedding_model):
        # NOTE need error handling for collection not found, but add Collection class to handle everything
        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

        # use vector store as a retriever

        # override the default similarity search function to display scores
        @chain
        def retriever(query: str) -> List[Document]:
            docs, scores = zip(*vector_store.similarity_search_with_score(query, k=5))
            for doc, score in zip(docs, scores):
                doc.metadata["score"] = score

            return docs

        contexts = retriever.invoke(query)

        return contexts

    def retrieve_contexts_with_multi_queries(
        self, query, collection_name, embedding_model, llm
    ):
        # NOTE need error handling for collection not found, but add Collection class to handle everything

        # query expansion: create multiple queries from original query with llm
        output_parser = LineListOutputParser()
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI psychological counselor. Your task is to generate five 
            different versions of the given user question to retrieve relevant documents from a vector
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        query_expansion_chain = QUERY_PROMPT | llm | output_parser

        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

        multi_query_retriever = MultiQueryRetriever(
            retriever=vector_store.as_retriever(),
            llm_chain=query_expansion_chain,
            parser_key="lines",
        )  # "lines" is the key (attribute name) of the parsed output

        contexts = multi_query_retriever.invoke(query)

        # deduplicate the documents
        unique_documents = set()
        for context in contexts:
            unique_documents.add(document_to_str(context))

        final_contexts = [str_to_document(doc) for doc in unique_documents]
        print(f"Retrieved Contexts: {len(final_contexts)}")

        return final_contexts

    # Generation ---------------------------------------------------------------
    def create_tool_with_collection(self, collection_name, embedding_model):
        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

        tool = create_retriever_tool(
            vector_store.as_retriever(),
            "cbt_knowledge_retriever",
            "Searches and returns information about cognitive behavioral therapy.",
        )

        return tool

    def create_rag_chain_tool_with_collection(
        self, collection_name, embedding_model, llm
    ):
        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

        system_prompt = """
            You are an assistant for question-answering tasks.
            Use the below context to answer the question. If
            you don't know the answer, say you don't know.
            Use three sentences maximum and keep the answer
            concise.

            Question: {question}

            Context: {context}
            """
        # prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])

        prompt = hub.pull("rlm/rag-prompt")
        retriever = vector_store.as_retriever()
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG Chain Schema: ", rag_chain.input_schema.schema())

        rag_tool = rag_chain.as_tool(
            name="cbt_knowledge_expert",
            description="Generates concise answers to questions about cognitive behavioral therapy.",
        )

        return rag_tool

    # helpful getter functions -----------------------------------------------
    def reset_manager(self):
        """
        Reset database manager
        """
        self.client.reset()

    def get_collection_names(self):
        """
        Get collection names
        """
        return [collection.name for collection in self.client.list_collections()]

    # def __del__(self):
    #     self.client.reset()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
