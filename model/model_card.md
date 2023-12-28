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

The Graph below shows the FPR bias on features, pay attention to the attribute with high FPR. By using this model, you may face inaccurate
result when predict high educated person, like bacheloars-school/doctor degree etc.

![bias-graph](/model/fpr_fiarness_graph.png)

## Caveats and Recommendations

The model itself is still in development. The recall and precision can reach higher score by applying a more fine-grained data cleaning and engineering step. And in current data engineering step, we handle native-country as a categorical features which makes the data become sparse, pre-divide the native-country into a class represent the GDP of that country may also help to imporve the model's ability.