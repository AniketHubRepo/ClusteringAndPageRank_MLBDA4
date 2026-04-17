# M25DE1051 | Aniket Srivastava | Assignment 4 | Part 1: Clustering
# CSL7110: Machine Learning with Big Data

import numpy as np
import time
import random
import os


# Task 1: Read feature vectors from text file
def readVectorsSeq(filename):
    """Read comma-separated vectors; last column is class label and is dropped."""
    vectors = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = list(map(float, line.split(',')))
            vectors.append(np.array(vals[:-1], dtype=np.float64))
    return vectors


# Task 2: Farthest First Traversal - k-Center Clustering
def kcenter(P, k):
    """
    k-Center via Farthest First Traversal.
    O(|P| * k) time - selects k centers greedily from P.
    """
    n = len(P)
    first_idx = random.randint(0, n - 1)
    centers = [P[first_idx]]
    min_dists = np.array([np.sum((p - centers[0]) ** 2) for p in P], dtype=np.float64)

    for _ in range(k - 1):
        farthest_idx = int(np.argmax(min_dists))
        new_c = P[farthest_idx]
        centers.append(new_c)
        new_dists = np.array([np.sum((p - new_c) ** 2) for p in P], dtype=np.float64)
        min_dists = np.minimum(min_dists, new_dists)

    return centers


# Task 3: k-Means++ Seeding Algorithm
def kmeansPP(P, k):
    """
    k-Means++ probabilistic center selection.
    O(|P| * k) time - samples centers with D^2 weighting.
    """
    n = len(P)
    first_idx = random.randint(0, n - 1)
    centers = [P[first_idx]]
    dists = np.array([np.sum((p - centers[0]) ** 2) for p in P], dtype=np.float64)

    for _ in range(k - 1):
        total = dists.sum()
        if total == 0:
            idx = random.randint(0, n - 1)
        else:
            probs = dists / total
            idx = int(np.random.choice(n, p=probs))
        new_c = P[idx]
        centers.append(new_c)
        new_dists = np.array([np.sum((p - new_c) ** 2) for p in P], dtype=np.float64)
        dists = np.minimum(dists, new_dists)

    return centers


# Task 4: k-Means Objective - Average Squared Distance
def kmeansObj(P, C):
    """
    Computes average squared distance of each point in P to its nearest center in C.
    Returns kmeans objective / |P|.
    """
    C_arr = np.array(C)
    total = 0.0
    for p in P:
        sq_dists = np.sum((C_arr - p) ** 2, axis=1)
        total += sq_dists.min()
    return total / len(P)


# Main program
def main():
    DATA_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'Q1', 'spambase.data')
    )

    k  = 10   # target cluster count
    k1 = 100  # coreset size (k < k1)

    print("=" * 60)
    print("CSL7110 Assignment 4  |  Part 1: Clustering")
    print(f"Dataset : spambase.data   k = {k}   k1 = {k1}")
    print("=" * 60)

    P = readVectorsSeq(DATA_PATH)
    print(f"\nDataset loaded: {len(P)} points, {len(P[0])} dimensions\n")

    # Step 1: kcenter(P, k) - print running time
    print(f"[Step 1]  kcenter(P, k={k})")
    t0 = time.time()
    C_kc = kcenter(P, k)
    elapsed_kc = time.time() - t0
    print(f"  Running time          : {elapsed_kc:.4f} s")
    obj1 = kmeansObj(P, C_kc)
    print(f"  kmeansObj (kcenter)   : {obj1:.4f}")

    # Step 2: kmeansPP(P, k) then kmeansObj(P, C)
    print(f"\n[Step 2]  kmeansPP(P, k={k}) then kmeansObj")
    t0 = time.time()
    C_kpp = kmeansPP(P, k)
    elapsed_kpp = time.time() - t0
    obj2 = kmeansObj(P, C_kpp)
    print(f"  Running time          : {elapsed_kpp:.4f} s")
    print(f"  kmeansObj(P, C)       : {obj2:.4f}")

    # Step 3: kcenter(P, k1) -> kmeansPP(X, k) -> kmeansObj(P, C)
    print(f"\n[Step 3]  kcenter(P, k1={k1}) -> kmeansPP(X, k={k}) -> kmeansObj")
    t0 = time.time()
    X = kcenter(P, k1)
    elapsed_x = time.time() - t0
    print(f"  kcenter(P, k1) time   : {elapsed_x:.4f} s  |  coreset |X| = {len(X)}")

    t0 = time.time()
    C_cs = kmeansPP(X, k)
    elapsed_cs = time.time() - t0
    obj3 = kmeansObj(P, C_cs)
    print(f"  kmeansPP(X, k) time   : {elapsed_cs:.4f} s")
    print(f"  kmeansObj(P, C)       : {obj3:.4f}")

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Step 1  kcenter(P, k)             kmeansObj = {obj1:.4f}")
    print(f"  Step 2  kmeansPP(P, k)            kmeansObj = {obj2:.4f}")
    print(f"  Step 3  coreset(k1={k1})+kmeans++   kmeansObj = {obj3:.4f}")
    print()
    print("  NOTE: kcenter provides a 2-approximation coreset.")
    print("  Larger k1 yields centers closer to direct kmeansPP on full P.")
    print()

    return {'obj1': obj1, 'obj2': obj2, 'obj3': obj3}


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    results = main()
