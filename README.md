# Node-Classification-GCN
Node Classification on Cora with Graph Convolutional Network (GCN)

This project demonstrates node classification on the Cora citation network dataset using a Graph Convolutional Network (GCN) implemented with PyTorch Geometric.

# Overview

Graph Neural Networks (GNNs) are powerful tools for learning from graph-structured data. In this project, we use a GCN to classify academic papers from the Cora dataset into their respective research topics by leveraging both node features and the citation graph structure.

# Features

Loads and preprocesses the Cora dataset automatically

Implements a two-layer GCN using PyTorch Geometric

Trains and evaluates the model with standard data splits

Visualizes training/validation loss and accuracy

Provides t-SNE visualization of learned node embeddings

Includes clear mathematical explanation of the GCN layer

# Dataset

The Cora dataset consists of:

2,708 scientific publications (nodes)

5,429 citation links (edges)

Each node is described by a 1,433-dimensional sparse feature vector

7 classes (research topics)

# Usage

Open the notebook (.ipynb) in Google Colab or locally in Jupyter.

Run all cells to train and evaluate the GCN on Cora.

Modify the code to experiment with other datasets or architectures (e.g., GAT, GraphSAGE).

# Mathematical Explanation

The **GCNConv** layer (Graph Convolutional Network Convolution) implements the following operation, as introduced by Kipf & Welling (2017):

$$
\mathbf{X}' = \hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-1/2} \mathbf{X} \mathbf{\Theta}
$$


Where:
1. $\mathbf{X}$ : Input node feature matrix (one row per node)
2. $\hat{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ : Adjacency matrix of the graph with added self-loops
3. $\hat{\mathbf{D}}$ : Diagonal degree matrix of $\hat{\mathbf{A}}$ (each diagonal element is the sum of the corresponding row of $\hat{\mathbf{A}}$)
4. $\mathbf{\Theta}$ : Trainable weight matrix (learned parameters)
5. $\hat{\mathbf{D}}^{-\frac{1}{2}} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-\frac{1}{2}}$ : Symmetric normalization to balance the influence of neighbors according to their degree

# References

Kipf, T.N. & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks

PyTorch Geometric Documentation

Cora Dataset Description
