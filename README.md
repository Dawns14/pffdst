# Parameter-Freezing-based Federated Dynamic Sparse Training (PFFDST)

## Dependencies
- Python 3.6 or greater
- PyTorch, torchvision
- tqdm

## Examples

| Experiment | Command line |
| ---------- | ------------ |
| FedAvg on CIFAR-10 | `python3 dst.py --dataset cifar10 --sparsity 0.0` |
| FedDST on CIFAR-10 (S=0.9) | `python3 dst.py --dataset cifar10 --sparsity 0.9 --rounds 400 --epochs 3` |
| FedDST+FedProx on CIFAR-10 (S=0.9, mu=1) | `python3 dst.py --dataset cifar10 --sparsity 0.9 --prox 1 --rounds 400 --epochs 3` |
| PFFDST on CIFAR-10 (S=0.95) | `python3 dst.py --dataset cifar10 --sparsity 0.95 --rounds 800 --freeze --epochs 2 --server-readjustment` |
| PFFDST w/o PF on CIFAR-10 (S=0.95) | `python3 dst.py --dataset cifar10 --sparsity 0.95 --rounds 400 --epochs 3 --server-readjustment` |
| PFFDST w/o SMR on CIFAR-10 (S=0.95) | `python3 dst.py --dataset cifar10 --sparsity 0.95 --rounds 800 --epochs 3 --freeze` |


## Acknowledgements
This work is developed based on the [FedDST]((https://github.com/bibikar/feddst.git) .
