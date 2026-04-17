# M25DE1051 | Aniket Srivastava | Assignment 4 | Part 3: PageRank on Spark
# CSL7110: Machine Learning with Big Data

import os
from pyspark import SparkContext, SparkConf


def load_edges(sc, filepath):
    """Load edge list from file, deduplicate, return RDD of (src, dst)."""
    raw = sc.textFile(filepath)
    edges = raw.map(lambda line: tuple(map(int, line.strip().split())))\
               .distinct()
    return edges


def build_adjacency(edges_rdd):
    """
    Build adjacency list RDD: (src, [dst1, dst2, ...]).
    Each src maps to its unique neighbors.
    """
    adj = edges_rdd.groupByKey().mapValues(lambda nbrs: list(set(nbrs)))
    return adj


def compute_out_degrees(adj_rdd):
    """Return RDD of (node, out_degree)."""
    return adj_rdd.mapValues(len)


def pagerank(sc, adj_rdd, n, beta=0.8, iterations=40):
    """
    Iterative PageRank with teleport probability (1 - beta).
    r_new = (1 - beta)/n * A + beta * M * r_old
    adj_rdd: (src, [dst, ...])
    n      : total number of nodes
    """
    teleport = (1.0 - beta) / n

    # Initial rank vector: r0 = 1/n for all nodes
    ranks = adj_rdd.mapValues(lambda _: 1.0 / n)

    for i in range(iterations):
        # Contributions: each src sends rank/out_degree to each neighbor
        contribs = adj_rdd.join(ranks).flatMap(
            lambda x: [(dst, x[1][1] / len(x[1][0])) for dst in x[1][0]]
        )
        # Sum contributions per node, apply teleport
        ranks = contribs.reduceByKey(lambda a, b: a + b)\
                        .mapValues(lambda s: teleport + beta * s)

        # Handle dangling nodes: nodes with no outgoing edges get teleport score
        # (adj_rdd only has nodes with out-edges, so re-join to include all)

    return ranks


def run_pagerank(graph_file, n, beta=0.8, iterations=40, label=""):
    """Run PageRank on a graph file and return top-5 and bottom-5 nodes."""
    conf = SparkConf().setAppName(f"PageRank_{label}").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    print(f"\n{'='*60}")
    print(f"PageRank: {label}  |  n={n}, beta={beta}, iterations={iterations}")
    print(f"{'='*60}")

    edges = load_edges(sc, graph_file)
    edge_count = edges.count()
    print(f"Edges (unique)  : {edge_count}")

    adj = build_adjacency(edges)
    node_count = adj.count()
    print(f"Nodes with edges: {node_count}")

    ranks = pagerank(sc, adj, n, beta=beta, iterations=iterations)
    ranked_list = ranks.collect()
    ranked_list.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 5 nodes (highest PageRank):")
    for rank_pos, (node, score) in enumerate(ranked_list[:5], 1):
        print(f"  #{rank_pos}  Node {node:5d}  score = {score:.6f}")

    print(f"\nBottom 5 nodes (lowest PageRank):")
    for rank_pos, (node, score) in enumerate(ranked_list[-5:], 1):
        print(f"  #{rank_pos}  Node {node:5d}  score = {score:.6f}")

    sc.stop()
    return ranked_list


def main():
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Q3'))
    SMALL  = os.path.join(BASE, 'small.txt')
    WHOLE  = os.path.join(BASE, 'whole.txt')

    print("=" * 60)
    print("CSL7110 Assignment 4  |  Part 3: PageRank on Spark")
    print("=" * 60)

    # Run on small graph first (n=53)
    small_ranks = run_pagerank(SMALL, n=53, beta=0.8, iterations=40, label="small.txt")
    top_small = small_ranks[0][1]
    print(f"\n  Verification: top score in small graph = {top_small:.6f}")
    print(f"  (Assignment reference: ~0.036)")

    # Run on full graph (n=1000)
    whole_ranks = run_pagerank(WHOLE, n=1000, beta=0.8, iterations=40, label="whole.txt")

    print("\n" + "=" * 60)
    print("Final Summary - whole.txt (n=1000)")
    print("=" * 60)
    print("\nTop 5 node IDs:")
    for node, score in whole_ranks[:5]:
        print(f"  Node {node}  score = {score:.6f}")
    print("\nBottom 5 node IDs:")
    for node, score in whole_ranks[-5:]:
        print(f"  Node {node}  score = {score:.6f}")
    print()


if __name__ == "__main__":
    main()
