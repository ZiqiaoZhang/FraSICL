# FraSICL
The source codes of FraSICL model.

paper: Molecular Property Prediction by Semantic-invariant Contrastive Learning

The current version of codes is just a proof-of-concept of FraSICL model. It is not runnable due to the lack of some requirements, e.g. an in-house training framework package and pre-training datasets.

Runnable scripts, pre-training dataset and pre-trained model checkpoints will be uploaded soon.

## Download pre-training dataset and model weights checkpoint
Our pre-training dataset and the weight checkpoint of pre-trained FraSICL model can be downloaded [here](https://drive.google.com/drive/folders/1NXHWMYWftYvwspHydPHzW6cTlzrc12Z0?usp=sharing).

pubchem-10m-clean.txt is the pre-train dataset of MolCLR, which contains ~10M molecules gathered from PubChem database.
200K samples are randomly selected from this dataset for curating our pre-train dataset, named pubchem-200K-screened.txt.

model-58-fin is the checkpoint of model weights for reproducing our baseline experimental results.


## ContrastiveFeaturizer.py
Codes for featurizing given molecules to input data for pretraining/finetuning a FraSICl model.


## ContrastiveModel.py
Codes for constructing a FraSICl model.

## ContrastiveEvaluator.py
An evaluator for estimating the performance of a FraSICL model.

## ContrastiveProcessController.py
A process controller which can handle the entire training/finetuning process for experiments.

## PretrainedDataPreProcessing.py
Scripts for generating pretrained dataset.

## Utils.py
Codes for calculating NT-Xent Loss.

