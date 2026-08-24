
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

| Parameter               | Type    | Default   | Description |
|-------------------------|--------|-----------|--------------|
| `dataset`               | str    | 'YelpChi' | Dataset name to use |
| `lr`                    | float  | 0.01      | Learning rate for optimizer |
| `weight_decay`          | float  | 0.0       | Weight decay for optimizer |
| `seed`                  | int    | 42        | Random seed for reproducibility |
| `hidden`                | int    | 64       | Hidden dimension size for embeddings |
| `order`                 | int    | 2         | Order of spectral filter in the model |
| `run`                   | int    | 1         | Run number (for multiple trials) |
| `epoch`                 | int    | 300       | Number of training epochs |
| `patience`              | int    | 30        | Early stopping patience |
| `gpu`                   | int    | 0         | GPU id to use |
| `alpha`                 | float  | 0.6       | Weight of r(ctr) |

---
## Configurations：

| Dataset | Hidden | Alpha | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| Aamzon-Full(homo) | 64 | 0.60 |  0.9305 ± 0.0070 | 0.7551 ± 0.0351 |
| Amazon | 64 | 0.60 |  0.9324 ± 0.0040 | 0.7544 ± 0.0183 |
| Amazon-Full | 128 | 0.50 |  0.9356 ± 0.0056 | 0.7790 ± 0.0132 |
| DGraph-Fin | 64 | 0.90 |  0.6648 ± 0.0121 | 0.0202 ± 0.0010 |
| Elliptic | 128 | 0.60 | 0.7259 ± 0.0209 | 0.1902 ± 0.0072 |
| T-Finance | 64 | 0.90 | 0.8522 ± 0.0069 | 0.6231 ± 0.0122 |
| Weibo | 64 | 0.90 | 0.9269 ± 0.0011 | 0.8108 ± 0.0056 |
| YelpChi | 128 | 0.60 |  0.7395 ± 0.0158 | 0.1324 ± 0.0089 |
| YelpChi-Full | 128 | 0.50 |  0.6591 ± 0.0121 | 0.2416 ± 0.0110 |
| YelpChi-Full(homo) | 64 | 0.30 |  0.6208 ± 0.0058 | 0.2188 ± 0.0080 |

To train model on a specific dataset, use the following command:

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
