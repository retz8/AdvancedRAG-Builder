# main testing file
import os
import argparse
from dotenv import load_dotenv

from RAGIndexing import RAGIndexing


load_dotenv()

if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set GOOGLE_API_KEY in .env file")

    rag_indexing = RAGIndexing(
        file_type="pdf",
        file_path="./data/sample/common_questions_cbt.pdf",
        database_type="vector",
        config=None,
    )
    # testing overall flow
    rag_indexing.test()
