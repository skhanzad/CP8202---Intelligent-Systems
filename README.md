# Toward Graph-Based Memory in Intelligent Systems

## Introduction

This project extends RAG with a **knowledge-graph-based memory system** that maintains a persistent, structured representation of everything discussed across a conversation. When a user sends a message, the system retrieves a semantically relevant subgraph and injects it as structured context before generating a response. After the response is generated, the user's message is processed by a second LLM call that extracts new nodes and edges and merges them into the graph.

A **forgetting module** runs periodically to prune low-importance nodes, preventing the graph from growing unboundedly and keeping retrieval focused on what is most relevant.

The system is built in Python using **LangGraph** for the main flow, **Gemma2 via Ollama** as the LLM, and **nomic-embed-text via Ollama** for embeddings. Evaluation is performed on the **LoCoMo** and **LongMemEval** benchmarks, which test long-horizon conversational memory and reasoning.
