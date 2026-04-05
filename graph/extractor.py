import json
import re

import requests

from graph.embedder import get_embedding, cosine_similarity
from graph.graph_manager import (
    add_edge,
    add_node,
    find_node_by_label,
    merge_node,
)

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "gemma2"
EMBED_SIM_THRESHOLD = 0.97

_SYSTEM_PROMPT = """\
You are a knowledge graph extractor. Given a user message, extract all entities, \
concepts, and events as a structured JSON object.

## Node classification

**semantic** — a persistent, named entity or concept: any person, organisation, \
location, technology, role, academic domain, or product that could be referenced \
again in a future message.

**episodic** — a specific, time-bound event or experience explicitly described \
in this message (something that happened at a particular time or place).

## Rules

1. Every named entity — person, organisation, location, technology, role, concept, \
or domain — is its own node. Never absorb a named entity into another node or \
collapse distinct entities together.
2. Attributes are reserved for atomic scalar properties that cannot stand alone \
as entities: a numeric value, a date, a single-word category tag. \
If a value could itself have relationships with other entities, it must be a node \
with an edge, not an attribute.
3. When a concrete, time-bound activity is described, extract a dedicated episodic \
node for it and connect all participants, locations, and topics to that event node \
with appropriate edges. Named entities involved remain their own semantic nodes.
4. If the message carries no meaningful entities, concepts, or events \
(e.g. greetings, filler phrases, acknowledgements), output empty lists: \
{"nodes": [], "edges": []}.
5. Always include "User" (type: semantic) as the speaker node, unless rule 4 applies.
6. Relation labels must be UPPER_SNAKE_CASE verb phrases.
7. If the same entity appears under different surface forms, use one canonical label.
8. Output ONLY valid JSON — no prose, no markdown fences.

## Output schema

{"nodes": [{"label": "<string>", "type": "semantic|episodic", "attributes": {}}],
 "edges": [{"source": "<label>", "target": "<label>", "relation": "<RELATION>"}]}

## Examples

### Example 1 — semantic entities only

Input: "My colleague Bob works at DeepMind in London as a research scientist."

Output:
{"nodes": [
  {"label": "User",              "type": "semantic", "attributes": {}},
  {"label": "Bob",               "type": "semantic", "attributes": {}},
  {"label": "DeepMind",          "type": "semantic", "attributes": {}},
  {"label": "London",            "type": "semantic", "attributes": {}},
  {"label": "Research Scientist", "type": "semantic", "attributes": {}}
 ],
 "edges": [
  {"source": "User", "target": "Bob",               "relation": "KNOWS"},
  {"source": "Bob",  "target": "DeepMind",          "relation": "WORKS_AT"},
  {"source": "Bob",  "target": "Research Scientist", "relation": "HAS_ROLE"},
  {"source": "DeepMind", "target": "London",        "relation": "LOCATED_IN"}
 ]
}

### Example 2 — semantic and episodic nodes together

Input: "Last Tuesday I attended a guest lecture on computer vision at McGill University."

Output:
{"nodes": [
  {"label": "User",                                      "type": "semantic", "attributes": {}},
  {"label": "Computer Vision",                           "type": "semantic", "attributes": {}},
  {"label": "McGill University",                         "type": "semantic", "attributes": {}},
  {"label": "Attended guest lecture on computer vision", "type": "episodic", "attributes": {"when": "last Tuesday"}}
 ],
 "edges": [
  {"source": "User", "target": "Attended guest lecture on computer vision", "relation": "PARTICIPATED_IN"},
  {"source": "Attended guest lecture on computer vision", "target": "Computer Vision",   "relation": "COVERED_TOPIC"},
  {"source": "Attended guest lecture on computer vision", "target": "McGill University", "relation": "TOOK_PLACE_AT"}
 ]
}

### Example 3 — no meaningful content

Input: "Okay, sounds good!"

Output:
{"nodes": [], "edges": []}"""


def _node_text(label: str, attributes: dict, sentence_context: str = "") -> str:
    parts = [label]
    if attributes:
        attr_str = ", ".join(f"{k}: {v}" for k, v in attributes.items())
        parts.append(attr_str)
    result = ": ".join(parts) if len(parts) > 1 else parts[0]
    if sentence_context:
        result = f"{result} — {sentence_context}"
    return result


def _strip_fences(text: str) -> str:
    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)

    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    text = text[start : i + 1]
                    break

    text = re.sub(r",\s*([\]\}])", r"\1", text)
    return text.strip()


def extract(user_input: str) -> dict:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        "stream": False,
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"]
    return json.loads(_strip_fences(raw))


def merge_extraction_into_graph(
    graph: dict, extraction: dict, user_input: str = ""
) -> dict:
    label_to_id: dict[str, str] = {}

    # Pre-collect all existing embeddings once to avoid re-fetching per node
    existing_embeddings: list[tuple[str, list[float]]] = [
        (node_id, node["embedding"])
        for node_id, node in graph["nodes"].items()
        if node.get("embedding")
    ]

    for node_spec in extraction.get("nodes", []):
        label: str = node_spec["label"]
        ntype: str = node_spec.get("type", "semantic")
        attributes: dict = node_spec.get("attributes", {})

        # Stage 1 — exact label match
        existing_id = find_node_by_label(graph, label)
        if existing_id is not None:
            merge_node(graph, existing_id, attributes)
            label_to_id[label] = existing_id
            continue

        # Stage 2 — embedding similarity with sentence context.
        embed_text = _node_text(label, attributes, user_input)
        label_embedding = get_embedding(embed_text)
        best_id: str | None = None
        best_sim: float = 0.0
        for node_id, emb in existing_embeddings:
            sim = cosine_similarity(label_embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_id = node_id

        if best_id is not None and best_sim > EMBED_SIM_THRESHOLD:
            merge_node(graph, best_id, attributes)
            label_to_id[label] = best_id
        else:
            new_id = add_node(graph, label, ntype, attributes, label_embedding)
            label_to_id[label] = new_id

    for edge_spec in extraction.get("edges", []):
        src_label: str = edge_spec["source"]
        tgt_label: str = edge_spec["target"]
        relation: str = edge_spec["relation"]

        src_id = label_to_id.get(src_label)
        tgt_id = label_to_id.get(tgt_label)
        if src_id and tgt_id:
            add_edge(graph, src_id, tgt_id, relation)

    return label_to_id