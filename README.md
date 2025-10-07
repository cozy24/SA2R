# AND-ONE

AND-ONE is a self-supervised framework that jointly exploits structural, attribute, and perturbation signals for multi-relational graph anomaly detection.

---

## Datasets

Overview of eight datasets used in our experiments. The table shows the number of nodes (and anomalies), anomaly ratio, number of relations, number of edges, and number of attributes for each dataset.

| Dataset       | Nodes (Anomalies) | Anomaly Ratio | Relations                       | Edges                         | Attributes |
|---------------|-----------------|---------------|---------------------------------|-------------------------------|------------|
| Weibo         | 8,405 (868)      | 10.3%         | -                               | 815,926                       | 400        |
| Amazon        | 10,224 (693)     | 6.8%          | U--P--U                         | 351,216                       | 25         |
| YelpChi       | 23,831 (1,217)   | 5.1%          | R--U--R                         | 98,630                        | 32         |
| T-Finance     | 39,357 (1,803)   | 4.6%          | -                               | 42,484,443                    | 10         |
| Elliptic      | 203,769 (4,545)  | 2.2%          | -                               | 234,355                       | 166        |
| DGraph-Fin    | 3,700,550 (15,509)| 0.4%       | -                               | 4,300,999                     | 17         |
| Amazon-Full   | 11,944 (821)     | 6.9%          | U--P--U, U--S--U, U--V--U       | 351,216, 7,132,958, 2,073,474| 25         |
| YelpChi-Full  | 45,954 (6,677)   | 14.5%         | R--S--R, R--T--R, R--U--R       | 6,805,486, 1,147,232, 98,630 | 32         |

**Note on dataset access:**  
- T-Finance can be downloaded following [BWGNN](https://github.com/squareRoot3/Rethinking-Anomaly-Detection).
- Elliptic and DGraph-Fin datasets are downloaded and preprocessed following [GADBench](https://github.com/squareRoot3/GADBench).  

---

## Parameters

Key parameters for running AND-ONE:

| Parameter               | Type    | Default   | Description |
|-------------------------|--------|-----------|-------------|
| `print_results`         | bool   | True      | Whether to print graph info |
| `dataset`               | str    | 'YelpChi' | Dataset name to use |
| `lr`                    | float  | 0.01      | Learning rate for optimizer |
| `weight_decay`          | float  | 0.0       | Weight decay for optimizer |
| `seed`                  | int    | 42        | Random seed for reproducibility |
| `batch_size_sampling`   | int    | 262144    | Batch size used during neighbor sampling |
| `hidden`                | int    | 128       | Hidden dimension size for embeddings |
| `order`                 | int    | 2         | Order of spectral filter in the model |
| `run`                   | int    | 1         | Run number (for multiple trials) |
| `epoch`                 | int    | 300       | Number of training epochs |
| `patience`              | int    | 20        | Early stopping patience |
| `eval_epoch`            | int    | 10        | Evaluate the model every n epochs |
| `gpu`                   | int    | 0         | GPU id to use |
| `lamda`                 | float  | 1         | Weight of noise MSE loss |

---

## Running AND-ONE

To train AND-ONE on a specific dataset, use the following command:

python run.py --dataset <DATASET> --lr <LEARNING_RATE> --hidden <HIDDEN_DIM> --epoch <NUM_EPOCHS> --gpu <GPU_ID>

| Dataset Flag            | Description                                                            |
| ----------------------- | ---------------------------------------------------------------------- |
| `--dataset yelp`        | Multi-relational YelpChi-Full dataset.                                 |
| `--dataset amazon`      | Multi-relational Amazon-Full dataset.                                  |
| `--dataset YelpChiFull` | Convert YelpChi-Full into a homogeneous graph (all edge types merged). |
| `--dataset AmazonFull`  | Convert Amazon-Full into a homogeneous graph (all edge types merged).  |
| `--dataset weibo`       |                                                                        |
| `--dataset elliptic`    |                                                                        |
| `--dataset tfinance`    |                                                                        |
| `--dataset dgraphfin`   |                                                                        |
