# main testing file
import os
import yaml
from dotenv import load_dotenv

from RAGIndexing import RAGIndexing

from AdvancedRAG import AdvancedRAG

load_dotenv()


def load_yaml_config(config_file_path):
    """Loads YAML configuration from a file, handling potential errors.

    @args:
      config_file_path: The path to the YAML configuration file.

    @returns:
      The parsed YAML data as a Python dictionary, or None if an error occurs.
    """

    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            return config_data
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    return None


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Please set GOOGLE_API_KEY in .env file")

    # load configuration file
    config_file = "config.yaml"
    config = load_yaml_config(config_file)

    if config:
        # initialize AdvancedRAG with configuration
        advanced_rag_manager = AdvancedRAG(config=config)
        # debug
        # print(advanced_rag_manager.get_dataset_names())
        advanced_rag_manager.create_indexing(dataset_name="sample-collection")

        # del advanced_rag_manager

    else:
        exit()

    # rag_indexing = RAGIndexing(
    #     file_type="pdf",
    #     file_path="./data/sample/common_questions_cbt.pdf",
    #     database_type="vector",
    #     config=None,
    # )
    # # testing overall flow
    # rag_indexing.test()
