# knowledge-graph-learning
Use of GNN architectures for prediction tasks on knowledge graphs. 

We tested our models on DBPEDIA, with self-loops and isolated nodes removal. 

## Link Prediction

We used a two layer R-GCN to obtain knowledge graph node embeddings, then DistMult as decoder for link prediction task. Nodes have Local Degree Profile (LDP) and one-hot encoded types as features. 

We tested hetero graph learning models. In particular, HeteroGAT outperforms R-GCN models.
