import os
from typing import TypedDict

import chromadb

from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, START, END

from fastapi import FastAPI
from pydantic import BaseModel

from prompts import build_rag_prompt

#configuration

MOCK_LLM = os.getenv("MOCK_LLM", "1")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#load the sentence transformer
print("Loading embedding model")
embedding_model=SentenceTransformer(
    EMBEDDING_MODEL
)

#now connect to chromadb
print("connecting to chromadb")
client=chromadb.PersistentClient(
    path=CHROMA_PATH
)
collection=client.get_collection(
    name=COLLECTION_NAME
)
print("Chromadb collection",collection.name)
print("number of stored chunks",collection.count())

#creating pydantic model for fastapi requests

#creating pydantic request model
class AskRequest(BaseModel):
    query:str
#creating pydantic response model
class AnswerResponse(BaseModel):
    answer:str
    sources:list[str]
    confidence:float

#Creating state of langgraph

class GraphState(TypedDict,total=False):
    query:str
    intent:str
    retrieved_chunks:list[str]
    retrieved_ids:list[str]
    top_chunk_snippet:str
    answer:str
    sources:list[str]
    confidence:float

#creating node1-classify intent

def classify_intent(state:GraphState):
    query=state["query"]
    lowered_query=query.lower()
    policy_keywords=["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

    #mock llm
    if MOCK_LLM=="1":
        if any(keyword in lowered_query for keyword in policy_keywords):
            intent="policy_question"
        else:
            intent="general_question"
    print(f"[classify_intent] {query}->{intent}")
    return {"intent":intent}

#creating node2- retrieve and answer

def retrieve_and_answer(state:GraphState):
    query=state["query"]
    print("[retrieve_and_answer] Query:",query)

    #embed user query
    query_embedding=embedding_model.encode(query).tolist()

    #retrieve top 3 chunks from chromadb
    results=collection.query(
        query_embeddings=[query_embedding],n_results=3
    )

    documents=results["documents"][0]
    ids=results["ids"][0]
    print(
        "[retrieve_and_answer] Retrieved:",len(documents),"chunks"
    )
    if not documents:
        return {
            "retrieved_chunks":[],
            "retrieved_ids":[],
            "top_chunk_snippet":"",
            "answer":(
                "The requested information could not be"
                "found in the provided documents."
            ),
            "sources":[],
            "confidence":0.0
        }
    #top chunk
    top_chunk=documents[0]
    top_chunk_snippet=top_chunk[:200]

    #build context
    context="\n\n".join(documents)

    #build rag prompt
    prompt=build_rag_prompt(
        question=query,
        context=context
    )

    #print prompt while development
    print("\n[RAG_PROMPT]")
    print(prompt)
    print("[END RAG_PROMPT]\n")

    #mock llm
    if MOCK_LLM=="1":
        answer=(
            "Based on the retrieved context:"
            f"{top_chunk_snippet}"
        )
        confidence=1.0

    #return state
    return {
        "retrieved_chunks":documents,
        "retrieved_ids":ids,
        "top_chunk_snippet":top_chunk_snippet,
        "answer":answer,
        "sources":ids,
        "confidence":confidence
    }

#creating node 3-direct_answer
def direct_answer(state:GraphState):
    query=state["query"]
    print(
        "[direct_answer] Query:",
        query
    )
    #mock llm
    if MOCK_LLM=="1":
        answer=(
            "I can answer questions about Zepto policies" \
            "right now."
        )
        confidence=1.0
    return {
        "answer":answer,
        "sources":[],
        "confidence":confidence
    }

#conditional routing
def route_intent(state:GraphState):
    if state["intent"]=="policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

#Build langgraph
workflow=StateGraph(GraphState)

#adding three nodes

workflow.add_node("classify_intent",classify_intent)
workflow.add_node("retrieve_and_answer",retrieve_and_answer)
workflow.add_node("direct_answer",direct_answer)

#adding start edge
workflow.add_edge(START,"classify_intent")
#adding conditional edge
workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer":"retrieve_and_answer",
        "direct_answer":"direct_answer"
    }
)
#adding end edge
workflow.add_edge("retrieve_and_answer",END)
workflow.add_edge("direct_answer",END)
#Compile graph
graph=workflow.compile()

#=========================================================
# FastAPI
#=========================================================

app=FastAPI(title="Zepto Support Assistant")

#=========================================================
#Post/ask
#=========================================================

@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask(request:AskRequest):
    result=graph.invoke(
        {
            "query":request.query
        }
    )

    response=AnswerResponse(
        answer=result["answer"],
        sources=result.get("sources",[]),
        confidence=result.get("confidence",1.0)
    )
    return response


