import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and index a database")
    parser.add_argument('dataset_path', type=str, help="Path to the dataset")
    parser.add_argument('database_type', type=str, choices=['vector', 'graph'], help="Type of the database (vector or graph)")
    
    args = parser.parse_args()

    print(args.dataset_path)