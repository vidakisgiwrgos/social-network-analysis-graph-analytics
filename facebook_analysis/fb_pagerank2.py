import os
import networkx as nx

dataset_dir = r"C:\Users\User\Videos\sn_project\facebook"
G = nx.Graph()
for fname in os.listdir(dataset_dir):
    if fname.endswith(".edges"):
        with open(os.path.join(dataset_dir, fname)) as f:
            for line in f:
                u, v = map(int, line.split())
                G.add_edge(u, v)

# Για ορισμένα metrics χρειαζόμαστε directed view
D = G.to_directed()

# Degree Centrality
deg_cent = nx.degree_centrality(G)

# Betweenness Centrality
btw_cent = nx.betweenness_centrality(G, normalized=True)

# Closeness Centrality
clo_cent = nx.closeness_centrality(G)

# Eigenvector Centrality
eig_cent = nx.eigenvector_centrality(G, max_iter=100, tol=1e-06)

# PageRank
pr = nx.pagerank(D, alpha=0.85, tol=1e-06)

# Συνάρτηση για top-k
def top_k(cent_dict, k=10):
    return sorted(cent_dict.items(), key=lambda x: x[1], reverse=True)[:k]

# Εκτύπωση Top-10 για κάθε μέτρο
metrics = {
    "Degree": deg_cent,
    "Betweenness": btw_cent,
    "Closeness": clo_cent,
    "Eigenvector": eig_cent,
    "PageRank": pr
}

for name, cent in metrics.items():
    print(f"\n=== Top-10 by {name} Centrality ===")
    for rank, (node, score) in enumerate(top_k(cent), start=1):
        print(f"{rank:2d}. User {node:<6} → {score:.6f}")
