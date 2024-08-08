# main testing file
import argparse

from RAGIndexing import RAGIndexing


if __name__ == "__main__":
    rag_indexing = RAGIndexing()
    # testing overall flow
    rag_indexing.test()