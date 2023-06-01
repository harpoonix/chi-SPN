# The Causal Loss: A Naïve Causal-Neural Connection

# Abstract

Most algorithms in classical and contemporary machine learning are trained to estimate features in correlation-based settings. Although success has been observed in many relevant problems, these algorithms are not guaranteed to perform well when the model is either oblivious to the underlying causal structure or inconsistent with its implied causal relations. Recent works have proposed different parameterizations of causal models in the Pearlian notion of causality, one of them being based on Probabilistic Circuits (PCs). In this work we show that by leveraging PCs as density estimators with tractable inference properties we can teach classical and modern machine learning models to estimate causal quantities. Specifically, we demonstrate training on neural networks and use common explainable AI methods to reveal the correct transfer of causal relations during our training procedure.

## Structure of the repository

* All figures of the paper can be found in the `figures` folder.
* Scripts for training and evaluating experiments are placed under `ciSPN`.
  * Note: Run all scripts from the folders that they are located in.
* Results are stored in the `experiments` folder.
* Summaries and tables can be created from scripts located under `ciSPN/evals` and `figures`.


## Datasets

This repository contains code and short descriptions two novel datasets presented in the paper. Scripts for creating all datasets and the actual data itself are stored in the `datasets` folder. A brief description, the causal graph and structural equations of the Hidden Object Data set and Causal Health Classification Data set are provided as pdf
* Causal Health Classification Dataset: [PDF](./docs/CausalHealthClassification.pdf)
* Hidden Object Dataset: [PDF](./docs/HiddenObjectDataset.pdf)

## Decision Trees

Decision Trees for all data set of Experiment 3 can be viewed in the following PDF: [DecisionTreeModels.pdf](./docs/DecisionTreeModels.pdf).

## GradCAM

Additional GradCAM visualizations of CNN trained with MSE (left), ciSPN trained with NLL (middle) and CNN trained with ciSPN (right) on the Hidden Object data set. CiSPN and NN+causal loss utilize the provided intervention information and subsequently focus the causally correct objects.
![Grad Cam Images](./figures/gradCAMAppendix/figure_appendix_gradCAMCausalImg.jpg "Grad CAM Images of models trained on the Hidden Object Dataset.")

