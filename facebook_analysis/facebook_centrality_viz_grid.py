import os
import networkx as nx
import matplotlib.pyplot as plt

dataset_dir = r"C:\Users\User\Videos\sn_project\facebook"

# Φόρτωση undirected γράφου
G = nx.Graph()
for fname in os.listdir(dataset_dir):
    if fname.endswith(".edges"):
        with open(os.path.join(dataset_dir, fname), "r") as f:
            for line in f:
                u, v = map(int, line.split())
                G.add_edge(u, v)

print(f"Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Μετατροπή σε directed για PageRank
D = G.to_directed()

# Υπολογισμός κεντρικοτήτων
deg_cent = nx.degree_centrality(G)
btw_cent = nx.betweenness_centrality(G, normalized=True)
clo_cent = nx.closeness_centrality(G)
eig_cent = nx.eigenvector_centrality(G, max_iter=100, tol=1e-06)
pr       = nx.pagerank(D, alpha=0.85, tol=1e-06)

# Συνάρτηση Top-10
def top_k(d, k=10):
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

metrics = {
    "Degree":    deg_cent,
    "Betweenness": btw_cent,
    "Closeness":   clo_cent,
    "Eigenvector": eig_cent,
    "PageRank":    pr,
}

# 2×2 Bar‐charts για τα πρώτα 4 measures
fig, axes = plt.subplots(2, 2, figsize=(12,10))
axes = axes.flatten()
for ax, (name, cent) in zip(axes, list(metrics.items())[:4]):
    top10 = top_k(cent)
    nodes, scores = zip(*top10)
    ax.bar(range(len(nodes)), scores)
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels(nodes, rotation=45, ha='right')
    ax.set_title(f"Top 10 by {name} Centrality")
    ax.set_ylabel("Score")
fig.tight_layout()
plt.show()

# Bar‐chart για PageRank ξεχωριστά
top10_pr = top_k(pr)
nodes, scores = zip(*top10_pr)
plt.figure(figsize=(8,4))
plt.bar(range(len(nodes)), scores)
plt.xticks(range(len(nodes)), nodes, rotation=45, ha='right')
plt.title("Top 10 by PageRank")
plt.ylabel("PageRank Score")
plt.tight_layout()
plt.show()

# Scatter plots: PageRank vs τα άλλα 3 κεντρικά μέτρα
pairs = [
    ("Degree",       deg_cent),
    ("Betweenness",  btw_cent),
    ("Closeness",    clo_cent),
    ("Eigenvector",  eig_cent),
]
fig, axes = plt.subplots(1, len(pairs), figsize=(5*len(pairs),4))
for ax, (name, cent) in zip(axes, pairs):
    ax.scatter(list(pr.values()), list(cent.values()), alpha=0.3)
    ax.set_xlabel("PageRank Score")
    ax.set_ylabel(f"{name} Centrality")
    ax.set_title(f"PageRank vs {name}")
fig.tight_layout()
plt.show()
