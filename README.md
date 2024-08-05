# AdvancedRAG-Builder
A flexible Framework for building advanced Retrieval-Augmented Generation (RAG) applications. AdvancedRAG-Builder supports integration with various databases, including vector databases and knowledge graphs. It offers features like context reranking, compression, and query expansion to enhance the quality of responses. Ideal for developers looking to create customizable and scalable RAG solutions for diverse datasets.

## Folder Structure
- data: contains the data files used (formats: .csv, .pdf, .txt)
- indexing: [RAG Step 1]:
     load document, process document, split document, and create index(vector or graph) with chunks

## Dependencies
- argparse: handle command-line arguments easily