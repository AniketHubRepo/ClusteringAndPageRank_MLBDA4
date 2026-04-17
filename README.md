# CSL7110 – Machine Learning with Big Data  
## Assignment 4: Clustering & PageRank  

## Student Details  
- Name: Aniket Srivastava  
- Roll No: M25DE1051  
- Course: CSL7110 – Machine Learning with Big Data  
- Assignment: 4  

## Project Overview  
This assignment implements clustering, inverted index search engine, and PageRank using Python and PySpark.

## Repository Structure  
CSL7110_Assignment4/   
├── Part1_Clustering/    
├── Part2_WebSearch/    
├── Part3_PageRank/    
├── data/    
└── README.md    

## Part 1: Clustering  
Implemented:
- Farthest First Traversal (k-center)
- k-Means++ initialization  

Key Functions:
- readVectorsSeq
- kcenter
- kmeansPP
- kmeansObj  

## Part 2: Web Search – Inverted Index  
Implemented a search engine using inverted index supporting:
- addPage
- queryFindPagesWhichContainWord
- queryFindPositionsOfWordInAPage  

## Part 3: PageRank on Spark  
- Implemented using PySpark RDDs  
- Beta = 0.8, Iterations = 40  

## How to Run  
pip install pyspark numpy  

python Part1_Clustering/clustering.py  
python Part2_WebSearch/web_search.py  
python Part3_PageRank/pagerank.py  

## 🔗 GitHub  
https://github.com/AniketHubRepo/ClusteringAndPageRank_MLBDA4.git 

# Summary of Assignment 

This assignment provided a comprehensive understanding of three important domains in big data systems: clustering, information retrieval, and graph analytics.

In Part 1, it was observed that k-Means++ significantly outperforms the k-center approach when evaluated using the k-means objective function. This is because k-center focuses on minimizing the maximum distance, while k-means minimizes the average squared distance. The coreset-based approach highlighted the trade-off between computational efficiency and clustering quality.

In Part 2, the inverted index demonstrated how large-scale search systems efficiently retrieve relevant documents. The importance of preprocessing steps such as stop-word removal, normalization, and tokenization was clearly evident in improving query accuracy and performance.

In Part 3, the PageRank implementation showed how graph structure influences node importance. The algorithm converged efficiently due to the presence of a strongly connected graph. The variation in PageRank scores highlighted how connectivity and incoming links impact ranking in real-world networks.

Overall, the assignment emphasizes:
- The importance of algorithm selection based on objective functions  
- Efficient data structures for scalable search systems  
- Distributed computing for handling large graphs  
- Trade-offs between accuracy and computational cost  

---