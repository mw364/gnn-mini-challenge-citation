# GNN Mini-Challenge: Citation Graph Node Classification

This repository hosts a mini-competition on semi-supervised node classification using a directed citation graph.

Participants are required to predict topic labels for unseen nodes using Graph Neural Networks implemented with DGL, following concepts covered in lectures 1.1–4.6, including graph types, message passing, batching, and neighbor or cluster sampling.

---

## Task Description

You are given a citation network where nodes represent documents and directed edges represent citation relationships. Each node is associated with a sparse feature vector.

The goal is to predict the topic label (target) for all unlabeled nodes in the test set.

---

## Provided Files

The dataset is provided in CSV format:

- data/train.csv  
  Labeled training nodes with features and target labels.  
  Columns: id, feat_1, ..., feat_d, target

- data/test.csv  
  Unlabeled test nodes with features only.  
  Columns: id, feat_1, ..., feat_d

- data/edges.csv  
  Directed edges defining the citation graph.  
  Columns: src, dst

---

## Objective Metric

Submissions are evaluated using the Macro F1-score, defined as the unweighted mean of the per-class F1 scores.

This metric is used because the topic classes are imbalanced and Macro F1 measures performance fairly across all classes.

---

## Constraints

The following constraints apply to all submissions:

- Only the provided CSV files may be used. External datasets or pretraining are not allowed.
- The complete training and validation process must finish within 10 minutes on CPU.
- Models must be implemented using DGL-based GNN architectures as covered in lectures 1.1–4.6.
- Allowed techniques include GCN-style layers, neighbor or cluster sampling, batching, dropout, and early stopping.
- The total number of trainable parameters should be approximately 2 million or fewer.

---

