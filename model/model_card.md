# Model Card

## Model Details

This is a DecisionTreeClassifier from scikit-learn and trained on census dataset.
Trained on scikit-learn=1.3.2

## Intended Use

This model can be used to predict the salary of a people.

## Training Data

This model trained on census dataset [dataset](../data/raw_data/census.csv)

## Evaluation Data

when training, we split dataset into train, test first with test_size=0.2, and
then do KFold CrossValidation on model, KFold=5

## Metrics
precision: 0.748
recall: 0.557
fbeta: 0.638

## Ethical Considerations

The Graph below shows the FPR bias on features 

![bias-graph](/model/fpr_fiarness_graph.png)
