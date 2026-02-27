import os
import gzip
import pandas as pd
import networkx as nx
from collections import defaultdict
from community import community_louvain   # pip install python-louvain
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

raw_dir    = r"C:\Users\User\Videos\sn_project\ogbn-arxiv\ogbn-arxiv\ogbn_arxiv\raw"
edges_path = os.path.join(raw_dir, "edge.csv.gz")
year_path  = os.path.join(raw_dir, "node_year.csv.gz")   # file with one year per line

for p in (edges_path, year_path):
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {p}")

# Load citation edges (comma‐delimited)
edges = []
with gzip.open(edges_path, "rt") as f:
    for line in f:
        u_str, v_str = line.strip().split(",")
        edges.append((int(u_str), int(v_str)))
print(f"1) Loaded edges: {len(edges):,}")

# Load publication years (one year per node index)
year_map = {}
with gzip.open(year_path, "rt") as f:
    for idx, line in enumerate(f):
        s = line.strip().strip("'\"")  # remove quotes if present
        if not s:
            continue
        try:
            year_map[idx] = int(s)
        except ValueError:
            continue
print(f"2) Loaded year metadata for {len(year_map):,} papers")

# Determine snapshot years
years = sorted(set(year_map.values()))
if not years:
    raise RuntimeError("No publication years loaded.")
print(f"3) Snapshot years: {years[0]} … {years[-1]} ({len(years)} total)")

# Build yearly snapshots
snapshots = []
for yr in years:
    G = nx.Graph()
    for u, v in edges:
        if year_map.get(u, 0) <= yr and year_map.get(v, 0) <= yr:
            G.add_edge(u, v)
    snapshots.append((yr, G))
    print(f"   Year {yr}: nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")

# Static community detection per snapshot
partitions  = []
communities = []
for yr, G in snapshots:
    if G.number_of_edges() == 0:
        partitions.append((yr, {}))
        communities.append((yr, {}))
        print(f"   → Year {yr}: no edges, skipped")
        continue
    part = community_louvain.best_partition(G)
    partitions.append((yr, part))
    comms = defaultdict(set)
    for node, cid in part.items():
        comms[cid].add(node)
    communities.append((yr, comms))
    print(f"   → Year {yr}: detected {len(comms)} communities")

# Dynamic event detection
def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0

θ_survive, θ_merge = 0.5, 0.5
θ_split,   θ_birth = 0.5, 0.2
θ_death            = 0.2

events = []
for i in range(len(communities) - 1):
    yr_t,  comm_t   = communities[i]
    yr_t1, comm_t1  = communities[i+1]
    if not comm_t or not comm_t1:
        continue

    overlaps = {
        (c_t, c_t1): jaccard(comm_t[c_t], comm_t1[c_t1])
        for c_t in comm_t for c_t1 in comm_t1
    }

    # Survive
    for c_t in comm_t:
        best, score = max(((c_t1, overlaps[(c_t, c_t1)]) for c_t1 in comm_t1),
                          key=lambda x: x[1])
        if score >= θ_survive:
            events.append((yr_t1, "Survive", c_t, best, round(score,3)))
    # Merge
    for c_t1 in comm_t1:
        merged = [c_t for c_t in comm_t if overlaps[(c_t, c_t1)] >= θ_merge]
        if len(merged) > 1:
            detail = {c: round(overlaps[(c, c_t1)],3) for c in merged}
            events.append((yr_t1, "Merge", tuple(merged), c_t1, detail))
    # Split
    for c_t in comm_t:
        splits = [c_t1 for c_t1 in comm_t1 if overlaps[(c_t, c_t1)] >= θ_split]
        if len(splits) > 1:
            detail = {c: round(overlaps[(c_t, c)],3) for c in splits}
            events.append((yr_t1, "Split", c_t, tuple(splits), detail))
    # Death
    for c_t in comm_t:
        max_ov = max(overlaps[(c_t, c)] for c in comm_t1)
        if max_ov < θ_death:
            events.append((yr_t1, "Death", c_t, None, round(max_ov,3)))
    # Birth
    for c_t1 in comm_t1:
        max_ov = max(overlaps[(c, c_t1)] for c in comm_t)
        if max_ov < θ_birth:
            events.append((yr_t1, "Birth", c_t1, None, round(max_ov,3)))

# Display first events
events_df = pd.DataFrame(events, columns=["Year","Event","SourceComm","TargetComm","Score"])
print("\nDynamic Community Events (first 20):")
print(events_df.head(20))

# Plot modularity over years
yrs_plot, mods = [], []
for (yr, part), (_, G) in zip(partitions, snapshots):
    if G.number_of_edges() > 0:
        yrs_plot.append(yr)
        mods.append(community_louvain.modularity(part, G))
plt.figure(figsize=(6,4))
plt.plot(yrs_plot, mods, marker='o')
plt.title("Modularity over Years")
plt.xlabel("Year")
plt.ylabel("Modularity")
plt.tight_layout()
plt.show()

# Plot NMI between consecutive partitions
yrs_nmi, nmis = [], []
for idx in range(len(partitions)-1):
    yr1, p1 = partitions[idx+1]
    _,  p0  = partitions[idx]
    if p0 and p1:
        common = set(p0) & set(p1)
        labs0  = [p0[n] for n in common]
        labs1  = [p1[n] for n in common]
        yrs_nmi.append(yr1)
        nmis.append(normalized_mutual_info_score(labs0, labs1))
plt.figure(figsize=(6,4))
plt.plot(yrs_nmi, nmis, marker='s')
plt.title("NMI Between Consecutive Years")
plt.xlabel("Year Transition To")
plt.ylabel("NMI Score")
plt.tight_layout()
plt.show()
