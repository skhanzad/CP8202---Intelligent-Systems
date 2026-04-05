# Toward Graph-Based Memory in Intelligent Systems

## Overview
This project proposes a structured long-term memory framework for LLM-based agents using knowledge graphs augmented with a forgetting mechanism.

Modern LLM agents rely on long-term memory to maintain context across interactions. The dominant approach—retrieval-augmented generation (RAG)—stores information as independent vector embeddings. While scalable, this design struggles with relational reasoning, noise sensitivity, and maintaining coherent knowledge over time.

This work introduces a graph-based alternative where memory is modeled as a knowledge graph with explicit control over memory retention.

## Core Idea
Memories are represented as:
- **Nodes**: entities, events, or concepts  
- **Edges**: relationships between them  

The system maintains two memory types:
- **Episodic memory**: specific interactions (time-dependent)  
- **Semantic memory**: generalized knowledge derived from interactions  

## Architecture
The system operates in three stages:

1. **Retrieval**  
   Select relevant subgraphs based on current input.

2. **Update**  
   Extract entities and relationships from new data and integrate them into the graph.

3. **Retention (Forgetting Module)**  
   Remove low-utility memories using an **importance score** (e.g., recency, access frequency).