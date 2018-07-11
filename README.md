### frog

Port of [DARTS](https://github.com/quark0/darts) neural architecture search. 

Uses [basenet](https://github.com/bkj/basenet) as a lightweight pytorch wrapper.

#### Models

##### frog/models/CIFAR

Replication of the CIFAR10 experiment from original DARTS implementation.

The search phase produces exactly the same results as the original implementation.  

The final training phase has not been fully implemented yet -- the original implementation has some bells and whistles that are probably necessary to get (near) SOTA performance.  There are also some minor intentional difference from DARTS `train.py` (for simplicity purposes).  See [this issue](https://github.com/quark0/darts/issues/10) for some details.


##### frog/models/fashion_mnist

Tiny example that shows DARTS outperforming random search.  Should run in just a few minutes on a GPU.

#### Disclaimer

__Code under active development -- open an issue and I'll help as best I can__
