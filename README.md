# Drug Molecular Toxicity Prediction Using Neural Network

This project is to build a convolutional neural network (CNN) to predict the toxicity of a drug based on its molecular structure.

### Prerequisites

TensorFlow: 1.5.0  
TensorBoard (Optional): 1.5.1

### Dataset

This project uses a dataset of drug molecules in Simplified Molecular-Input Line-Entry System (SMILES) expressions and the binary labels indicating whether one drug molecule is toxic or not. The dataset is related to the toxicity of some small molecules. There are two folders, one is the training data NR-ER-train (~8k samples), another is the testing data NR-ER-test (~100 samples). The molecules in NR-ER-train and NR-ER-test do not overlap with each other. There are three files in each folder:

| File              | Type     | Description                                                                                                                                                                                                                                                                                               |
|-------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| names_smiles.csv  | String   | A *csv* file, each line contains a drug molecule’s name and its SMILES expression, separated by a comma (,)                                                                                                                                                                                                   |
| names_labels.csv  | String   | A *csv* file, each line contains a drug molecule’s name and its toxicity label, where 0 means non-toxic and 1 means toxic, separated by a comma (,)                                                                                                                                                           |
| names_onehots.npy | Numberic | A *npy* file which derived from names_smiles.csv, storing one-hot representations of SMILES expressions of drug molecules and can be loaded by numpy package, storing two ndarray; one is the names of the molecules, and the other is the one-hot representations of SMILES expressions of drug molecules |

## Running

Run [train.py](https://github.com/KirosC/dnn-toxicity-prediction/blob/master/train.py) to train a network model  
Run [evaluate.py](https://github.com/KirosC/dnn-toxicity-prediction/blob/master/evaluate.py) to test all the generated models in current path



## Built With

* [TensorFlow](https://www.tensorflow.org/) - The framework used
* [NumPy](http://www.numpy.org/) - Data operations

## Authors

* **Kiros Choi** - *Initial work* - [KirosC](https://github.com/KirosC)