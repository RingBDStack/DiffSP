# DiffSP

### Code Structure Overview

* **`checkpoint`**: This directory contains the trained models.
* **`data`**: Designated for storing the datasets used in the experiments.
* **`config`**: Contains the configuration files, which define the hyperparameters for training and testing.
* **`g`**: Includes the code for training and testing on the graph classification task.
* **`n`**: Includes the code for training and testing on the node classification task.
* **`model`**: Houses the implementation of the DiffSP model.
* **`utils`**: Contains the code for graph transfer entropy and node LID estimation.

### Data Download

To facilitate the reproduction of our experimental results, we provide the adversarial data used in our experiments.

Due to repository space limitations, the data is available at: [https://huggingface.co/datasets/anonymous-random/DiffSP-data/tree/main](https://huggingface.co/datasets/Mutual/DiffSP-data).

Please download the data and place it in the `./data/` directory before running the experiments.

### Run The Code

#### Node classification

For example, we run the experiments on the `Cora` dataset under the `PR-BCD` attack.

```
python n/main.py --dataset Cora --attack prbcd
```

If you want to retrain the DiffSP, use `--is_train` like:

```
python n/main.py --dataset Cora --attack prbcd --is_train
```

#### Graph classification

For example, we run the experiments on the `IMDB-BINARY` dataset under the `PR-BCD` attack.

```
python g/main.py --dataset IMDB-BINARY --attack prbcd
```

If you want to retrain the DiffSP, use `--is_train` like:

```
python g/main.py --dataset IMDB-BINARY --attack prbcd --is_train
```

#### Note

For graph classification, the classifier comprises two GCN layers, followed by a mean pooling layer and a linear layer, consistent across all models and identical to the surrogate model.

For node classification, the classifier consists of two GCN layers, consistent across all models and aligned with the surrogate model.
