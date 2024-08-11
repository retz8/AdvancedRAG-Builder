from langgraph.prebuilt import create_react_agent  # for langgraph agents
from langchain_core.messages import HumanMessage

from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import LongContextReorder


from .config import Config
from .database import VectorDBManager, KnowledgeGraphManager
from .file_processor import FileProcessor

from IPython.display import Image, display

from typing import Literal

from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


import sys  # for streaming


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

        # LLM
        # llm is for query expansion and other techniques
        # chat_llm is for main chatbot
        self.llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        self.chat_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", temperature=0
        )  # temperature=0 for deterministic output
        # can add safety settings later
        # https://ai.google.dev/api/python/google/generativeai/types/SafetySettingDict

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

    # NOTE Below codes are for testing, not used in the main code
    # Retrieval --------------------------------------------------------------
    def retrieve_contexts(self, query, dataset_name):
        # error handling
        print("==== Retrieving Contexts ====")
        dataset_type = self.config.get_dataset_type(dataset_name)
        if dataset_type == "vector":
            # retrieve from vector database
            embedding_model = self.config.get_dataset_embedding_model(dataset_name)

            # Retrieve Contexts (Naive)
            contexts = self.vector_db_manager.retrieve_contexts(
                query=query,
                collection_name=dataset_name,
                embedding_model=embedding_model,
            )

            # Retrieve Contexts (Multi Query Expansion)
            # contexts = self.vector_db_manager.retrieve_contexts_with_multi_queries(
            #     query=query,
            #     collection_name=dataset_name,
            #     embedding_model=embedding_model,
            #     llm=self.llm,
            # )
            print("\n")
            print(f"'{len(contexts)}' Retrieved contexts:")

            pretty_print_docs(contexts)

            # Re-Ranking Contexts
            # print("\n")
            # print("Re-ranked Retrieved contexts:")
            # reordered_contexts = self.rerank_contexts(contexts)

            # pretty_print_docs(reordered_contexts)

        # TODO: Implement retrieval for graph database
        elif dataset_type == "graph":
            pass

        else:
            print("Dataset type not supported.")
            exit(1)

    # Querying ---------------------------------------------------------------
    def test_query(self, query):
        # Let's add some prompt
        # print(self.llm.invoke(query))
        template = """Question: {question}

        Answer: Let's think step by step."""
        prompt = PromptTemplate.from_template(template)

        # 이런식으로 prompt를 llm이랑 엮는듯 (prompt의 결과를 llm에 넣는식)
        chain = prompt | self.llm

        # Let's stream the output as well
        # print(chain.invoke({"question": query}))

        for chunk in chain.stream({"question": query}):
            sys.stdout.write(chunk)
            sys.stdout.flush()

    def test_query_with_tools(self, query, dataset_name):
        embedding_model = self.config.get_dataset_embedding_model(dataset_name)

        # tool = self.vector_db_manager.create_tool_with_collection(
        #     collection_name=dataset_name, embedding_model=embedding_model
        # )
        # tools = [tool]
        # tool_node = ToolNode(tools)
        # model_with_tools = self.chat_llm.bind_tools(tool)

        system_message = "Your are a friendly counselor specialized in psychology and Cognitive Behavioral Therapy (CBT). But also, you may use some tools to explain about CBT."

        tool = self.vector_db_manager.create_rag_chain_tool_with_collection(
            collection_name=dataset_name,
            embedding_model=embedding_model,
            llm=self.chat_llm,
        )

        tools = [tool]

        # workflow = StateGraph(MessagesState)

        # def call_model(state: MessagesState):
        #     messages = state["messages"]
        #     response = model_with_tools.invoke(messages)
        #     return {"messages": [response]}

        # workflow.add_node("agent", call_model)
        # workflow.add_node("tools", tool_node)

        # workflow.add_edge("__start__", "agent")
        # workflow.add_conditional_edges(
        #     "agent",
        #     should_continue,
        # )
        # workflow.add_edge("tools", "agent")

        # app = workflow.compile()

        # try:
        #     display(Image(app.get_graph().draw_mermaid_png()))
        # except Exception:
        #     # This requires some extra dependencies and is optional
        #     pass

        # agent_executor = create_react_agent(
        #     self.chat_llm, tools, state_modifier=system_message
        # )
        # messages = agent_executor.stream(
        #     {"messages": [("user", query)]}, stream_mode="values"
        # )

        messages = [
            ("human", query),
        ]

        agent = create_react_agent(self.chat_llm, tools)

        # display(Image(agent_executor.get_graph().draw_mermaid_png()))

        # for chunk in agent.stream(
        #     {"messages": [("human", query)]}, stream_mode="values"
        # ):
        for chunk in agent.stream({"messages": messages}, stream_mode="values"):
            chunk["messages"][-1].pretty_print()

    # Post-Retrieval Techniques ----------------------------------------------
    def rerank_contexts(self, contexts):
        """
        Re-rank contexts to avoid "lost in the middle" problem

        @args:
        - contexts: list of contexts (Documents)

        @returns:
        - reordered_contexts: re-ordered contexts, more relevant contexts at the beginning
        """
        print("==== Re-Ranking ====")
        # more relevant contexts should be at the beginning
        reordering = LongContextReorder()
        reordered_contexts = reordering.transform_documents(contexts)

        return reordered_contexts

    # NOTE context compressions are not implemented yet


# Helper Functions -------------------------------------------------------
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        print(message)
        # if isinstance(message, tuple):
        #     print(message)
        # else:
        #     message.pretty_print()


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"
