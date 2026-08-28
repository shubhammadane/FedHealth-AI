# Methodology

## Preprocessing
- Duplicate removal
- Identifier removal (`encounter_id`, `patient_nbr`)
- Median imputation for numerical variables
- Most-frequent imputation for categorical variables
- One-hot encoding
- Standard scaling
- Stratified 80/20 train/test split

## Federated simulation

Five simulated clients are created from training data. IID mode shuffles before equal partitioning. Non-IID mode creates controlled statistical heterogeneity by separating target-sorted partitions. This is an experimental simulation and is not a claim about actual hospital populations.

## Models

Centralized:
- Logistic Regression
- Random Forest

Federated:
- Feed-forward PyTorch neural network
- FedAvg
- FedProx option

## Evaluation

Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, and convergence curves are generated from actual experiment outputs.
