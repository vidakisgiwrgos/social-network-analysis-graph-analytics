import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import community as community_louvain

dataset_dir = r"C:\Users\User\Videos\sn_project\facebook"

# Φόρτωση undirected γράφου
G = nx.Graph()
for fname in os.listdir(dataset_dir):
    if fname.endswith(".edges"):
        with open(os.path.join(dataset_dir, fname), "r") as f:
            for line in f:
                u, v = map(int, line.split())
                G.add_edge(u, v)

print(f"Φορτώθηκαν: {G.number_of_nodes():,} κόμβοι, {G.number_of_edges():,} ακμές")

# Louvain community detection
partition = community_louvain.best_partition(G)

# Μέτρα κοινοτήτων
comm_sizes = Counter(partition.values())
num_comms = len(comm_sizes)
print(f"Αριθμός κοινοτήτων: {num_comms}")

# Top-10 μεγαλύτερες κοινότητες
top10 = comm_sizes.most_common(10)
comm_ids, sizes = zip(*top10)
plt.figure(figsize=(8, 4))
plt.bar(range(len(sizes)), sizes)
plt.xticks(range(len(sizes)), [f"Comm {cid}" for cid in comm_ids], rotation=45, ha='right')
plt.title("Top 10 Κοινότητες κατά Μέγεθος (Louvain)")
plt.xlabel("Κοινότητα")
plt.ylabel("Αριθμός Κόμβων")
plt.tight_layout()
plt.show()

# Κατανομή μεγεθών όλων των κοινοτήτων
plt.figure(figsize=(6, 4))
plt.hist(list(comm_sizes.values()), bins=30)
plt.title("Κατανομή Μεγεθών Κοινοτήτων")
plt.xlabel("Μέγεθος Κοινότητας")
plt.ylabel("Πλήθος Κοινοτήτων")
plt.tight_layout()
plt.show()

# Οπτικοποίηση υπογράφου της μεγαλύτερης κοινότητας
largest_id = top10[0][0]
nodes_largest = [n for n, cid in partition.items() if cid == largest_id]
subG = G.subgraph(nodes_largest)

plt.figure(figsize=(6, 6))
pos = nx.spring_layout(subG, k=0.1, iterations=20)
nx.draw(subG, pos, node_size=20, node_color='skyblue', edge_color='gray', alpha=0.6)
plt.title(f"Υπογράφος Κοινότητας {largest_id}")
plt.axis('off')
plt.show()

# Δείγμα 2.000 κόμβων χρωματισμένο κατά κοινότητα
sample_nodes = list(G.nodes())[:2000]
subG_sample = G.subgraph(sample_nodes)
colors = [partition[n] for n in subG_sample.nodes()]

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(subG_sample, k=0.15)
nx.draw_networkx_nodes(subG_sample, pos, node_size=20, node_color=colors, cmap=plt.cm.tab20)
nx.draw_networkx_edges(subG_sample, pos, alpha=0.2)
plt.title("Δείγμα 2000 Κόμβων – Χρωματισμός κατά Κοινότητα")
plt.axis('off')
plt.show()
