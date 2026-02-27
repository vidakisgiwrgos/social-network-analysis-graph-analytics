import os
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, deque
import community as community_louvain  # pip install python-louvain

edges_path = r"C:\Users\User\Videos\sn_project\ogbn-arxiv\ogbn-arxiv\arxiv\raw\edge.csv.gz"

if not os.path.exists(edges_path):
    raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο:\n  {edges_path}")

# Φόρτωση του συμπιεσμένου CSV (gzip) σε DataFrame
df = pd.read_csv(
    edges_path,
    compression="gzip",
    header=None,
    names=["src", "dst"]
)
print(f"1) Loaded edge‐table: {len(df):,} γραμμές")

# Χτίσιμο πλήρους directed γράφου G_full
G_full = nx.DiGraph()
G_full.add_edges_from(df[["src", "dst"]].itertuples(index=False, name=None))
print(f"2) Full graph: {G_full.number_of_nodes():,} κόμβοι, {G_full.number_of_edges():,} ακμές")

# Snowball Sampling: BFS expansion για ~4.000 κόμβους
target_size = 4000

seed_node = random.choice(list(G_full.nodes()))
sample_nodes = set([seed_node])
queue = deque([seed_node])

while queue and len(sample_nodes) < target_size:
    current = queue.popleft()
    neighbors = set(G_full.successors(current)) | set(G_full.predecessors(current))
    for nbr in neighbors:
        if len(sample_nodes) >= target_size:
            break
        if nbr not in sample_nodes:
            sample_nodes.add(nbr)
            queue.append(nbr)

print(f"3) Collected {len(sample_nodes)} nodes via BFS‐snowball (target ~{target_size}).")

# Induced subgraph στο sample
G_directed = G_full.subgraph(sample_nodes).copy()
print(f"4) Sampled subgraph: {G_directed.number_of_nodes():,} κόμβοι, {G_directed.number_of_edges():,} ακμές")

# Οπτικοποίηση sampled subgraph (προαιρετικό)
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G_directed, k=0.15, iterations=20)
nx.draw(
    G_directed,
    pos,
    node_size=10,
    node_color="skyblue",
    edge_color="gray",
    alpha=0.4
)
plt.title("Sampled Subgraph (Directed) ~4000 nodes")
plt.axis("off")
plt.tight_layout()
plt.show()

# Μετατροπή σε Undirected προκειμένου να τρέξει ο Louvain
G = G_directed.to_undirected()
print(f"5) Converted to undirected: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Louvain Community Detection
# (η βιβλιοθήκη απαιτεί ένα non‐directed γράφο)
partition = community_louvain.best_partition(G)
num_comms = len(set(partition.values()))
print(f"6) Detected {num_comms} communities")

# Ανάλυση μεγεθών κοινότητας
comm_sizes = Counter(partition.values())  # {community_id: μέγεθος}
print(f"   Number of communities: {num_comms}")

# Top-10 μεγαλύτερες κοινότητες
top10_comms = comm_sizes.most_common(10)
print("   Top-10 communities (ID → μέγεθος):")
for cid, size in top10_comms:
    print(f"      Comm {cid:3d} → {size:,} κόμβοι")

# Bar chart Top-10 Κοινοτήτων κατά Μέγεθος
comm_ids, sizes = zip(*top10_comms)
plt.figure(figsize=(8, 4))
plt.bar(range(len(sizes)), sizes, color='steelblue')
plt.xticks(range(len(sizes)), [f"Comm {cid}" for cid in comm_ids], rotation=45, ha="right")
plt.title("Top-10 Κοινότητες κατά Μέγεθος (Louvain)")
plt.xlabel("Κοινότητα ID")
plt.ylabel("Αριθμός Κόμβων")
plt.tight_layout()
plt.show()

# Ιστόγραμμα Κατανομής Μεγεθών Κοινοτήτων
plt.figure(figsize=(6, 4))
plt.hist(list(comm_sizes.values()), bins=30, color='teal', edgecolor='black')
plt.title("Κατανομή Μεγεθών Κοινοτήτων")
plt.xlabel("Μέγεθος Κοινότητας (κόμβοι)")
plt.ylabel("Πλήθος Κοινοτήτων")
plt.tight_layout()
plt.show()

# Οπτικοποίηση της Μεγαλύτερης Κοινότητας
largest_comm_id = top10_comms[0][0]
nodes_largest = [n for n, cid in partition.items() if cid == largest_comm_id]
subG = G.subgraph(nodes_largest)

plt.figure(figsize=(6, 6))
pos_sub = nx.spring_layout(subG, k=0.15, iterations=20)
nx.draw(
    subG,
    pos_sub,
    node_size=20,
    node_color='skyblue',
    edge_color='gray',
    alpha=0.6
)
plt.title(f"Υπογράφος της Μεγαλύτερης Κοινότητας (Comm {largest_comm_id})")
plt.axis("off")
plt.tight_layout()
plt.show()

# Δείγμα 2000 Κόμβων – Χρωματισμός κατά Κοινότητα
sample_nodes2 = random.sample(list(G.nodes()), 2000)
subG2 = G.subgraph(sample_nodes2)
colors = [partition[n] for n in subG2.nodes()]

plt.figure(figsize=(8, 8))
pos2 = nx.spring_layout(subG2, k=0.20, iterations=20)
nx.draw_networkx_nodes(
    subG2,
    pos2,
    node_size=20,
    node_color=colors,
    cmap=plt.cm.tab20,
    alpha=0.8
)
nx.draw_networkx_edges(subG2, pos2, alpha=0.2, edge_color='gray')
plt.title("Δείγμα 2000 Κόμβων – Χρωματισμός κατά Κοινότητα")
plt.axis("off")
plt.tight_layout()
plt.show()
