#Architecture of the RAG pipeline
  1.Uploading Documents
  2.Chunking Documents
  3.getting embeddings
  4.storing them in chromadb
  5.User Query
  6.classify the intent of query
  7.conditional routing
  8.Retrieving the top 3 answers
  9.Pydantic response
  10.FastAPI
  11.Docker

#Task 1:
Eight documents are loaded in documents folder and divided into chunks(I have done per document chunk because of small size.)
Each chunk is embedded using sentencetransformers with all-MiniLM-L6-v2 model.
The embeddings with ids and document chunks are stored in a persistent chromaDB collection named "my_documents".
the whole code of this process is written in "task1.py" file

#prompt
"prompt.py" file contains
The rag prompt that follows the structure ROLE->CONTEXT->TASK->FORMAT->LENGTH
prompt contains
->negetive constraint
->few shot examples
->retrieved context
->the user's question

#"main.py" contains the complete main code
created a langgraph with typedDict with three nodes
1.classify_intent
2.retrieve_and_answer
3.direct_answer

In mock mode policy questions are identified using a deterministic keyword list given
[delivery", "return", "refund", "membership", "tracking", "cancel", "gift card",  "support hours"]

for policy questionsthe query is embedded using all-MiniLM-L6-v2.
the top three most similar chunks are retrieved from chromadb.

The baseline uses MOCK_LLM.
the default mode is MOCK_LLM=1
This mode requires no API key and makes no network calls.

FastAPI
The application exposes:
POST/ask
we try out the query in website
eg:{
    "query":"Can I cancel my order?"
}
the response is in this json format
{
    "answer":"Based on the retrieved context:....",
    "sources":[
        "document1.txt_chunk_0"
    ],
    "confidence":1.0
}

The application uses:
 -SentenceTransformers
 -all-MiniLM-L6-v2
 -ChromaDB
 -LangGraph
 -Pydantic
 -FastAPI
 -Docker