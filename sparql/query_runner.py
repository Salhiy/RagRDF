from rdflib import Graph

def load_rdf_graph(path="data/rdf_graph.ttl"):
    g = Graph()
    g.parse(path, format="ttl")
    return g

def run_sparql(query: str, graph: Graph):
    results = graph.query(query)
    return [str(row[0]) for row in results]