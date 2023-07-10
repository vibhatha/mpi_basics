# HPC with Machine Learning

We will be learning about a few machine learning algorithms in this section. 
Then we will study how to parallelize this workload.

## Pre-requisites

```bash
pip install pandas matplotlib scikit-learn
```

## Running KMeans

Make sure your current folder is `machine_learning`

Run the sequential version of the algorithm

```bash
python kmeans.py
```

Run the distributed version of the algorithm

```bash
mpirun -n 4 python3 distributed_kmeans.py
```