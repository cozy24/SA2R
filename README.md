# AND-ONE

AND-ONE is a self-supervised framework that jointly exploits structural, attribute, and perturbation signals for label-free graph anomaly detection.

---

## Datasets

| Dataset       | Nodes      | Anomaly Ratio | Edge  Types                     | Edges                         | Dim        |
|---------------|------------|---------------|---------------------------------|-------------------------------|------------|
| Weibo         | 8,405      | 10.3%         | -                               | 815,926                       | 400        |
| Amazon        | 10,224     | 6.8%          | U--P--U                         | 351,216                       | 25         |
| YelpChi       | 23,831     | 5.1%          | R--U--R                         | 98,630                        | 32         |
| T-Finance     | 39,357     | 4.6%          | -                               | 42,484,443                    | 10         |
| Elliptic      | 203,769    | 2.2%          | -                               | 234,355                       | 166        |
| DGraph-Fin    | 3,700,550  | 0.4%          | -                               | 4,300,999                     | 17         |
| Amazon-Full   | 11,944     | 6.9%          | U--P--U, U--S--U, U--V--U       | 351,216, 7,132,958, 2,073,474 | 25         |
| YelpChi-Full  | 45,954     | 14.5%         | R--S--R, R--T--R, R--U--R       | 6,805,486, 1,147,232, 98,630  | 32         |

**Note on dataset access:**  
- T-Finance can be downloaded following [BWGNN](https://github.com/squareRoot3/Rethinking-Anomaly-Detection).
- Elliptic and DGraph-Fin datasets can be downloaded and preprocessed following [GADBench](https://github.com/squareRoot3/GADBench).  

---

## Parameters

Key parameters for running AND-ONE:

| Parameter               | Type    | Default   | Description |
|-------------------------|--------|-----------|--------------|
| `dataset`               | str    | 'YelpChi' | Dataset name to use |
| `lr`                    | float  | 0.01      | Learning rate for optimizer |
| `weight_decay`          | float  | 0.0       | Weight decay for optimizer |
| `seed`                  | int    | 42        | Random seed for reproducibility |
| `hidden`                | int    | 128       | Hidden dimension size for embeddings |
| `order`                 | int    | 2         | Order of spectral filter in the model |
| `run`                   | int    | 1         | Run number (for multiple trials) |
| `epoch`                 | int    | 300       | Number of training epochs |
| `patience`              | int    | 20        | Early stopping patience |
| `gpu`                   | int    | 0         | GPU id to use |
| `alpha`                 | float  | 0.6       | Weight of r(ctr) |

---

## Running AND-ONE

To train AND-ONE on a specific dataset, use the following command:

python main.py --dataset <DATASET> --hidden <HIDDEN_DIM> --alpha <ALPHA>

| Dataset                 | Description             |
| ----------------------- | ------------------------|
| `--dataset YelpChi`     | for YelpChi             |
| `--dataset Amazon`      | for Amazon              |
| `--dataset yelp`        | for YelpChi-Full        |
| `--dataset amazon`      | for Amazon-Full         |
| `--dataset YelpChiFull` | for YelpChi-Full(homo)  |
| `--dataset AmazonFull`  | for Amazon-Full(homo)   |
| `--dataset weibo`       | for Weibo               |  
| `--dataset elliptic`    | for Elliptic            |                                             
| `--dataset tfinance`    | for T-Finance           |                                                       
| `--dataset dgraphfin`   | for DGraph-Fin          | 

---
