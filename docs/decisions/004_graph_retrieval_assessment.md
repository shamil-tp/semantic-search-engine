# ADR 004: Do not add graph retrieval

The shipped corpus is two public technical documents and has no entity or relationship labels. The four available relevance judgments are insufficient to demonstrate a graph-based gain, and a graph store would add an unsupported operational dependency. Graph retrieval is therefore omitted. Reconsider it only after the corpus contains traceable entities/relations and an evaluation set demonstrates a benefit over the lexical/dense baseline.
