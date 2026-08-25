
---
## Environment Requirements

SA2R has been tested with the following environment:

| Software | Version |
|----------|---------|
| Python | 3.12.3 |
| CUDA | 13.0 |
| PyTorch | 2.8.0 |
| DGL | 2.5 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| Scikit-learn | 1.6.1 |
| SymPy | 1.14.0 |
| Matplotlib | 3.10.3 |
| tqdm | 4.67.1 |

The required Python packages are provided in `requirements.txt`

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

## Key Parameters

| Parameter               | Type    | Default   | Description |
|-------------------------|--------|-----------|--------------|
| `dataset`               | str    | 'Amazon' | Dataset name to use |
| `lr`                    | float  | 0.01      | Learning rate for optimizer |
| `seed`                  | int    | 42        | Random seed for reproducibility |
| `hidden`                | int    | 64       | Hidden dimension size for embeddings |
| `order`                 | int    | 2         | Order of spectral filter in the model |
| `run`                   | int    | 1         | Run number (for multiple trials) |
| `epoch`                 | int    | 300       | Number of training epochs |
| `patience`              | int    | 30        | Early stopping patience |
| `gpu`                   | int    | 0         | GPU id to use |
| `alpha`                 | float  | 0.6       | Weight of r(ctr) |

## Train Model
To train model on a specific dataset, use the following command:

python main.py --dataset Amazon --hidden 64 --alpha 0.6

| Dataset                 | Description             |
| ----------------------- | ------------------------|
| `--dataset Amazon`      | for Amazon              |
| `--dataset YelpChi`     | for YelpChi             |
| `--dataset amazon`      | for Amazon-Full         |
| `--dataset yelp`        | for YelpChi-Full        |
| `--dataset AmazonFull`  | for Amazon-Full(homo)   |
| `--dataset YelpChiFull` | for YelpChi-Full(homo)  |
| `--dataset weibo`       | for Weibo               |  
| `--dataset tfinance`    | for T-Finance           |  
| `--dataset elliptic`    | for Elliptic            |                                                                                                 
| `--dataset dgraphfin`   | for DGraph-Fin          | 

---
