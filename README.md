# GDeR

Implementation of paper "GDeR: Safeguarding Efficiency, Balancing, and Robustness via Prototypical Graph Pruning"

## Requirements

```
torch==1.12.0
torch_geometric==2.5.2
networkx==3.1
numpy==1.24.4
ogb==1.3.6
scikit-learn==1.3.2
scipy==1.10.1
torch-scatter==2.0.9
torch-sparse==0.6.14
torcheval==0.0.7
pandas==2.0.3
pyparsing==3.1.2
pillow==10.2.0
tqdm==4.66.2
```

## Usage

**Datasets** Download the required dataset ( MUTAG/DHFR/ogbg-molhiv/ogbg-pcba/ZINC ) to `./datasets` .

**Hyper-parameter** Hyper-parameter configurations can be set in ` ./Configures.py` .

**Reproducibility** Below are commands to reproduce the results.

`PNA` + `DHFR`
```
python -m models.train_small --clst 0.0001 --sep 0.5 --model_name pna --lr 0.001 --batch_size 24 --weight_decay 0.0 --max_epochs 200 --retain_ratio 0.2 --pruning_epochs 40 --mlp_out_dim 0 --dataset_name DHFR
```

`GPS`+ `ogbg-molhiv`:

```
python -m models.train_mologbg --clst 0.0001 --sep 0.3 --model_name gps --lr 0.0005 --batch_size 128 --weight_decay 3e-6 --max_epochs 100 --retain_ratio 0.3 --pruning_epochs 10 --mlp_out_dim 1 --dataset_name ogbg-molhiv
```
