# FraSICL
The source codes of FraSICL model.

paper: Molecular Property Prediction by Semantic-invariant Contrastive Learning

The current version of codes is just a proof-of-concept of FraSICL model. It is not runnable due to the lack of some requirements, e.g. an in-house training framework package and pre-training datasets.

Runnable scripts, pre-training dataset and pre-trained model checkpoints will be uploaded soon.

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

