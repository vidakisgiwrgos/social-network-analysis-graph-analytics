import os
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

edges_path = r"C:\Users\User\Videos\sn_project\ogbn-arxiv\ogbn-arxiv\arxiv\raw\edge.csv.gz"

if not os.path.exists(edges_path):
    raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο:\n  {edges_path}")

# Φόρτωση του συμπιεσμένου CSV (gzip) σε DataFrame
df = pd.read_csv(edges_path,
                 compression="gzip",
                 header=None,
                 names=["src", "dst"])
print(f"1) Loaded edge‐table: {len(df):,} γραμμές")

# Χτίσιμο πλήρους directed γράφου G_full
G_full = nx.DiGraph()
G_full.add_edges_from(df[["src", "dst"]].itertuples(index=False, name=None))
print(f"2) Full graph: {G_full.number_of_nodes():,} κόμβοι, {G_full.number_of_edges():,} ακμές")

# Snowball Sampling: BFS expansion για ~4.000 κόμβους
target_size = 4000

# Επιλέγουμε τυχαία έναν seed κόμβο
seed_node = random.choice(list(G_full.nodes()))

# BFS/DFS expansion
sample_nodes = set([seed_node])
queue = deque([seed_node])

while queue and len(sample_nodes) < target_size:
    current = queue.popleft()
    # Περιλαμβάνουμε όλους τους γείτονες (εκείνους που τους "cites" και εκείνους που "cited by")
    neighbors = set(G_full.successors(current)) | set(G_full.predecessors(current))
    for nbr in neighbors:
        if len(sample_nodes) >= target_size:
            break
        if nbr not in sample_nodes:
            sample_nodes.add(nbr)
            queue.append(nbr)

print(f"3) Collected {len(sample_nodes)} nodes via BFS‐snowball (target ~{target_size}).")

# Induced subgraph
G = G_full.subgraph(sample_nodes).copy()
print(f"4) Sampled subgraph: {G.number_of_nodes():,} κόμβοι, {G.number_of_edges():,} ακμές")

# Οπτικοποίηση sampled subgraph (spring layout)
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw(G, pos,
        node_size=10,
        node_color="skyblue",
        edge_color="gray",
        alpha=0.5)
plt.title("Snowball‐Sampled Subgraph (~4000 nodes)")
plt.axis("off")
plt.tight_layout()
plt.show()

# Υπολογισμός κεντρικοτήτων στο sampled subgraph
deg_cent = nx.degree_centrality(G)
btw_cent = nx.betweenness_centrality(G, normalized=True)
clo_cent = nx.closeness_centrality(G)
eig_cent = nx.eigenvector_centrality(G, max_iter=100, tol=1e-06)
pr      = nx.pagerank(G, alpha=0.85, tol=1e-06, max_iter=100)

print("5) Υπολογίστηκαν όλα τα κεντρικά μέτρα (degree, betweenness, closeness, eigenvector, PageRank).")

# Συνάρτηση Top‐10
def top_k(centrality_dict, k=10):
    return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:k]

# Bar‐charts Top‐10 για Degree, Betweenness, Closeness, PageRank
metrics = {
    "Degree Centrality":       deg_cent,
    "Betweenness Centrality":  btw_cent,
    "Closeness Centrality":    clo_cent,
    "PageRank":                pr
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for ax, (name, cent_dict) in zip(axes, metrics.items()):
    top10 = top_k(cent_dict)
    nodes, scores = zip(*top10)
    ax.bar(range(len(nodes)), scores)
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels(nodes, rotation=45, ha="right")
    ax.set_title(f"Top 10 by {name}")
    ax.set_ylabel("Score")
fig.tight_layout()
plt.show()

# Bar‐chart Top‐10 για Eigenvector (ξεχωριστά)
top10_eig = top_k(eig_cent)
nodes_eig, scores_eig = zip(*top10_eig)
plt.figure(figsize=(8, 4))
plt.bar(range(len(nodes_eig)), scores_eig)
plt.xticks(range(len(nodes_eig)), nodes_eig, rotation=45, ha="right")
plt.title("Top 10 by Eigenvector Centrality")
plt.ylabel("Eigenvector Score")
plt.tight_layout()
plt.show()

# Histogram κατανομής PageRank Scores
plt.figure(figsize=(6, 4))
plt.hist(list(pr.values()), bins=50)
plt.title("Κατανομή PageRank Scores (Snowball‐Sampled)")
plt.xlabel("PageRank Score")
plt.ylabel("Πλήθος Κόμβων")
plt.tight_layout()
plt.show()

# Scatter plots: PageRank vs Degree/Betweenness/Closeness
pairs = [
    ("Degree",     deg_cent),
    ("Betweenness", btw_cent),
    ("Closeness",   clo_cent)
]
fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4))
for ax, (name, cent_dict) in zip(axes, pairs):
    ax.scatter(list(pr.values()), list(cent_dict.values()), alpha=0.3)
    ax.set_xlabel("PageRank Score")
    ax.set_ylabel(f"{name} Centrality")
    ax.set_title(f"PageRank vs {name}")
fig.tight_layout()
plt.show()

# Scatter plot: PageRank vs Eigenvector
plt.figure(figsize=(6, 4))
plt.scatter(list(pr.values()), list(eig_cent.values()), alpha=0.3)
plt.xlabel("PageRank Score")
plt.ylabel("Eigenvector Centrality")
plt.title("PageRank vs Eigenvector Centrality")
plt.tight_layout()
plt.show()
