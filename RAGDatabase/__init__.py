# RAGDatabase/__init__.py
import os

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    GoogleGenerativeAI,
    ChatGoogleGenerativeAI,
)
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, Tool, AgentExecutor

from langchain.tools.retriever import create_retriever_tool
from langchain import hub

import chromadb
from uuid import uuid4


class RAGDatabase:
    """
    LLM RAG Database (vector: chromaDB, graph: Neo4j)

    @args:
    - database_type: vector DB or knowledge graph (vector, graph)
    - embedding_model: text embedding model path (gemini api) (optional)
    """

    def __init__(self, database_type: str, embedding_model="models/text-embedding-004"):
        self.database_type = database_type
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model)

        if self.database_type == "vector":
            # initialize chroma db (for testing, we use persistent client)
            self.client = chromadb.PersistentClient()

    # testing with vector db first
    def init_db(self, collection_name):
        """
        Initialize database with empty collection
        """

        if self.database_type == "vector":
            # check if the collection already exists
            # if so, delete it first
            existing_collections = self.client.list_collections()
            if any(
                collection.name == collection_name
                for collection in existing_collections
            ):
                print(
                    f"Collection '{collection_name}' already exists. Deleting existing collection."
                )
                self.client.delete_collection(collection_name)

            # create a new collection with embedding function
            self.client.create_collection(collection_name, self.embedding_model)
            print(f"Created new collection '{collection_name}'")

            return

        # TODO: add grpah database support
        print("Database type not supported.")

    def add_docs_to_db(self, docs, collection_name):
        """
        Add documents to database

        @args
        - docs: list of documents (Langchain Document) to insert
        - collection_name: chromaDB collection (table) name
        """

        if self.database_type == "vector":
            # error handling: collection not found
            existing_collections = self.client.list_collections()
            if not any(
                collection.name == collection_name
                for collection in existing_collections
            ):
                print(
                    f"Collection '{collection_name}' not found. You need to initialize the database first."
                )
                return

            # get collection object (here, we are using langchain-chroma, not bare chromadb)
            vector_store = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embedding_model,
            )
            # insert documents
            # 1. obtain unique ids for each document
            uuids = [str(uuid4()) for _ in range(len(docs))]
            # 2. add documents to collection
            vector_store.add_documents(documents=docs, ids=uuids)
            print(f"Added `{len(docs)}` documents to collection '{collection_name}'")

            return

        # TODO: add grpah database support
        print("Database type not supported.")
        return

    def retrieve_contexts(self, query, collection_name):
        """
        Retrieve contexts from database

        @args:
        - query: input query
        - collection_name
        """
        print("==== Retrieving contexts with query `{query}` ====")
        if self.database_type == "vector":
            # query the vector database
            vector_store = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embedding_model,
            )

            # query the collection
            results = vector_store.similarity_search_by_vector(
                embedding=self.embedding_model.embed_query(query), k=5
            )
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

            return results

    def generate_response(self, query, collection_name):
        """
        Generate response

        @args:
        - query: input query
        - collection_name
        """
        print("==== Generating response with query ====")

        # initialize LLM
        llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            convert_system_message_to_human=True,
        )

        if self.database_type == "vector":
            vector_store = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embedding_model,
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})

            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise. Please provide the summary of context you used to answer at the end."
                "\n\n"
                "{context}"
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # generate response
            response = rag_chain.invoke({"input": query})
            print("LLM Response: ", response["output"])

            return rag_chain

    # TODO: Improve this method to be more flexible
    # need to change AgentExecutor to LangGraph
    def generate_response_with_tools(self, rag_chain, query, collection_name):
        """
        Generate response using langchain tools
        """
        # Currently, only one rag_chain is used as a tool
        print("==== Generating response with tools ====")

        # initialize LLM (need to use ChatGoogleGenerativeAI)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_output_tokens=100,
        )

        vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_model,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # define tools
        retrieval_tool = create_retriever_tool(
            retriever,
            name="cbt_general_knowledge",
            description="Useful when you need to answer questions about general knowledge on cognitive behavioral therapy (CBT).",
        )
        tools = [retrieval_tool]
        # tools = [
        #     Tool(
        #         name="CBT_QA",
        #         func=rag_chain.invoke,
        #         description="""Useful when you need to answer questions about general knowledge on cognitive behavioral therapy (CBT).
        # Use full question as input.
        # """,
        #     ),
        # ]

        # Get the prompt to use - you can modify this!
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # generate response
        response = agent_executor.invoke({"input": query})
        print(response["output"])

        return agent_executor

    def delete_db(self, collection_name):
        """
        Delete database
        """
        if self.database_type == "vector":
            # delete collection
            try:
                self.client.delete_collection(collection_name)
                print(f"Deleted collection '{collection_name}'")

            except Exception as e:
                print(f"Error deleting collection: {e}")
                print(self.client.list_collections())

    def text_to_embedding(self, text):
        """
        Convert text to embedding vector

        @args:
        - text: input text
        """
        vector = self.embedding_model.embed_query(text)
        return vector

    def reset_db(self):
        """
        Reset database clients
        """
        if self.database_type == "vector":
            print("==== Resetting Vector Database (ChromaDB) ====")
            self.client.reset()
            print("Database reset complete.")

    def list_collections(self):
        """
        List collections
        """
        if self.database_type == "vector":
            collections = self.client.list_collections()
            print(collections)
