# main testing file
import os
from dotenv import load_dotenv

from AdvancedRAG import AdvancedRAG

load_dotenv()


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set GOOGLE_API_KEY in .env file")

    # load configuration file
    config_file = "config.yaml"

    # initialize AdvancedRAG with configuration
    advanced_rag_manager = AdvancedRAG(config=config_file)
    # debug
    # print(advanced_rag_manager.get_dataset_names())

    # 1. create indexing for sample-collection
    advanced_rag_manager.create_indexing(dataset_name="sample-collection")

    # 2.

    # del advanced_rag_manager
