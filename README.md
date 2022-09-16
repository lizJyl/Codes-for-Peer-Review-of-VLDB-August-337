# Codes-for-Peer-Review-of-VLDB-August-337

## Prerequisites

- Python 3.6
- NVIDIA GPU V100
- Ubuntu
- PyTorch 1.8
- CUDA 10.1

## Run
``` 
sh start.sh
```

# data format with queries


### From Snape

x. circle   ground truth community from snap

x.edges     edges from snap

x.egofeat   feature of ego node (x) from snap

x.feat      features of other nodes   from snap



### Process by author


x.sample   the queries   each line is a query with one-hot format

x.labels   the community result of each query in x.sample  each line is a qeury result with one-hot format

x.att      the community attributes of each GT in x.circle  (for AFC setting)
