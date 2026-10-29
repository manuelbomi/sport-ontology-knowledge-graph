# NBA Fan Knowledge Graph

A hands-on walkthrough of knowledge graphs, ontologies, entity resolution, embeddings, and
retrieval-augmented generation (RAG) — using NBA fan data as the domain.

The NBA turned out to be a surprisingly good learning domain for this: fans follow teams that have
rosters that change every summer, players get traded, the same fan shows up as "steph_fan_94" on
the app and "Stephen A." in the CRM, and "engaged fan" means different things to different teams.
All the messy, real-world problems that make semantic data systems interesting.

---

## What's covered

| File | Concept |
|------|---------|
| `knowledge_graph.py` | Ontology, entities, relationships, graph traversal |
| `entity_resolution.py` | Linking "LBJ" and "LeBron James" to the same node |
| `embeddings.py` | Turning text into vectors; semantic search with FAISS |
| `rag_pipeline.py` | Grounding LLM answers in a graph + retrieved documents |
| `semantic_layer.py` | Defining business metrics consistently across systems |

---

## Core concepts explained

### Ontology

An ontology is a **blueprint for what kinds of things exist** in your domain and how they can
relate to each other. It's more than a schema — it captures *meaning*, not just structure.

```
ONTOLOGY (the blueprint)                KNOWLEDGE GRAPH (the populated instance)
────────────────────────────────────    ────────────────────────────────────────

  Thing                                   LeBron James   (Player)
  ├── Person                              Jayson Tatum   (Player)
  │   ├── Player  ──plays_for──> Team     Marcus         (Fan)
  │   └── Fan     ──follows──>   Team
  ├── Team        ──home_court──> Venue     Lakers ──home_court──> Crypto.com Arena
  ├── Venue       ──located_in──> City      Lakers ──plays_in──>   Los Angeles
  ├── City
  └── Event       ──hosted_at──> Venue
```

The ontology says "a Player *can* plays_for a Team." The knowledge graph says *which*
specific player plays for *which* specific team.

Python classes can represent an ontology (see `knowledge_graph.py`). Production systems use
OWL/RDF (W3C standard) or property graph schemas (Neo4j labels).

---

### Entities

An entity is a **specific, identifiable instance** of a type in your ontology.

```
Type: Player         Instances (entities):
                     ┌──────────────────────────────────────────┐
                     │  id: player_2544  name: LeBron James     │
                     │  id: player_1628  name: Jayson Tatum      │
                     │  id: player_201939 name: Stephen Curry   │
                     └──────────────────────────────────────────┘
```

Each entity has:
- A **canonical ID** (stable, unique across systems)
- **Attributes** (name, age, position)
- **Relationships** to other entities

---

### Entity Resolution

The same real-world entity often appears under many names in different systems:

```
  ticketing system:  "L. James"
  social media:      "KingJames"        →  resolve to →   entity: LeBron James
  box score feed:    "LeBron James"                        id: player_2544
  merch platform:    "lebron 23"
  CRM notes:         "king james lakers"
```

Entity resolution (also called *entity linking* or *record linkage*) is the process of
recognizing these as the same entity. See `entity_resolution.py` for implementations.

---

### Knowledge Graph

A knowledge graph stores **entities as nodes** and **relationships as edges**. It differs from
a relational database in that relationships are first-class citizens — you can traverse them
directly without JOINs.

```
                           FOLLOWS
        Marcus ────────────────────────────────► Lakers
          │                                         │
          │ LIVES_IN                                │ HOME_COURT
          ▼                                         ▼
       Los Angeles                         Crypto.com Arena
                                                    ▲
        Celtics ──────────────────────────────────── (hosted at)
                          PLAYS_AT            NBA Finals Game 2
```

Multi-hop query: *"Find all fans in LA whose team's home court hosted a Finals game"*
— that's three edge traversals, not a JOIN.

---

### Embeddings

Embeddings turn text (or structured data) into a **list of numbers — a vector** — where
*similar things end up close together* in the vector space.

```
  "LeBron wins championship ring"  →  [ 0.12, -0.87,  0.43, ...,  0.09 ]  ─┐
  "NBA Finals MVP ceremony"        →  [ 0.10, -0.91,  0.48, ...,  0.11 ]  ─┤  near each other
  "championship celebration party" →  [ 0.09, -0.88,  0.41, ...,  0.08 ]  ─┘

  "quarterly earnings report Q3"   →  [-0.73,  0.22, -0.31, ...,  0.54 ]     far away
```

This lets you search by *meaning* rather than exact keywords. See `embeddings.py`.

---

### Vector Search

Once you have embeddings, you can find the **nearest neighbors** to a query vector —
items that are semantically similar to the query.

```
  Query: "Who is the Warriors' best shooter?"

  distance  doc
  ──────────────────────────────────────────────────────────
  0.03      "Stephen Curry is known for elite 3-point shooting"
  0.11      "Golden State Warriors backcourt depth"
  0.19      "Curry career three-point record broken"
  0.72      "Lakers injury report: LeBron listed doubtful"  (far away)
```

FAISS (Facebook AI Similarity Search) is the standard library for this at scale.

---

### RAG — Retrieval-Augmented Generation

Instead of asking an LLM to answer from its training data alone (which can hallucinate or
be stale), you first **retrieve** relevant facts and then **generate** a grounded answer.

```
  Question: "What venue does Marcus's favorite team play at?"

  ┌─────────────────────────────────────────────────────────┐
  │ STEP 1: RETRIEVE                                        │
  │   graph lookup → Marcus follows Lakers                  │
  │   graph expand → Lakers home_court Crypto.com Arena     │
  │   vector search → "Lakers home arena" docs              │
  └─────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │ STEP 2: ASSEMBLE CONTEXT                                │
  │   "Marcus follows the Lakers.                           │
  │    The Lakers play at Crypto.com Arena."                │
  └─────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │ STEP 3: GENERATE (LLM with grounded context)            │
  │   Answer: "Crypto.com Arena"  ← factual, not guessed   │
  └─────────────────────────────────────────────────────────┘
```

See `rag_pipeline.py` for the full implementation.

---

### Semantic Layer

A semantic layer sits between raw data and downstream consumers (dashboards, ML models, LLMs)
and enforces **consistent definitions** of business concepts.

Without one, "engaged fan" might mean:
- 5+ game views (analytics team)
- 2+ purchases in 90 days (revenue team)
- Weekly app open (product team)

With a semantic layer, it's defined once:
```python
# everyone reads from the same definition
ENGAGED_FAN = (event_views >= 5) | (purchases_90d >= 2) | (weekly_app_opens >= 1)
```

The semantic layer also normalizes **metrics** (what counts as a "game view"?) and
**features** (how is "days_since_last_visit" computed?). See `semantic_layer.py`.

---

## The full system

```
  ┌──────────────────────────────────────────────────────┐
  │   RAW DATA: ticketing, app events, CRM notes, docs   │
  └──────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┴──────────────────┐
          ▼                                    ▼
  ┌────────────────┐                  ┌──────────────────┐
  │ ENTITY RESOL.  │                  │ EMBEDDING PIPE   │
  │ + Knowledge    │                  │ + Vector Index   │
  │   Graph        │                  │   (FAISS)        │
  └────────────────┘                  └──────────────────┘
          │                                    │
          └─────────────────┬──────────────────┘
                            ▼
                  ┌─────────────────┐
                  │ CONTEXT ASSEMBLY│  ← graph facts + retrieved docs
                  └─────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  LLM (grounded) │  ← RAG
                  └─────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │ ANSWER + TRACES │  ← explainability
                  └─────────────────┘
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

# run each module
python knowledge_graph.py      # builds and visualizes the graph
python entity_resolution.py    # entity linking examples
python embeddings.py           # vector search demo
python rag_pipeline.py         # full RAG walkthrough
python semantic_layer.py       # metric / feature definitions
```

The graph visualization saves to `knowledge_graph.png`. Everything else prints to stdout.

---

## Further reading

- [What Is a Knowledge Graph?](https://www.oreilly.com/library/view/knowledge-graphs/9781098127985/) — O'Reilly book, very practical
- [FAISS documentation](https://faiss.ai/) — Facebook's vector search library
- [Sentence Transformers](https://www.sbert.net/) — the embedding model used here
- [RAG paper (Lewis et al. 2020)](https://arxiv.org/abs/2005.11401) — original RAG paper
- [dbt semantic layer](https://docs.getdbt.com/docs/use-dbt-semantic-layer/dbt-sl) — production semantic layer example
