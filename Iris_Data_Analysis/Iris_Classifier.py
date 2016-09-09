#%matplotlib inline
import pandas as pd
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

# extract data
iris_data_clean = pd.read_csv('/Users/babburi/Documents/Classroom/DataScience/MyFirstRepo/iris-data_clean.csv')

# Testing the data: analysis will stop if any of the below assertions are wrong

# Classifier should have only three classes
assert len(iris_data_clean['class'].unique()) == 3

# sepal lengths for 'Iris-versicolor' should never be below 2.5 cm
assert iris_data_clean.loc[iris_data_clean['class'] == 'Iris-versicolor', 'sepal_length_cm'].min() >= 2.5

# data set should not have any missing measurements
assert len(iris_data_clean.loc[(iris_data_clean['sepal_length_cm'].isnull()) |
                              (iris_data_clean['sepal_width_cm'].isnull()) |
                              (iris_data_clean['petal_length_cm'].isnull()) |
                              (iris_data_clean['petal_width_cm'].isnull())]) == 0

all_inputs = iris_data_clean[['sepal_length_cm', 'sepal_width_cm',
                             'petal_length_cm', 'petal_width_cm']].values

all_classes = iris_data_clean['class'].values

# classifer
random_forest_classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                                 max_depth=None, max_features=3, max_leaf_nodes=None,
                                                 min_samples_leaf=1, min_samples_split=2,
                                                 min_weight_fraction_leaf=0.0, n_estimators=5, n_jobs=1,
                                                 oob_score=False, random_state=None, verbose=0, warm_start=True)
# plot the cross validation scores
rf_classifier_scores = cross_val_score(random_forest_classifier, all_inputs, all_classes, cv=10)
sb.boxplot(rf_classifier_scores)
sb.stripplot(rf_classifier_scores, jitter=True, color='white')

# output some predictions from the classifier
(training_inputs, testing_inputs,
training_classes, testing_classes) = train_test_split(all_inputs, all_classes, train_size=0.75)

random_forest_classifier.fit(training_inputs, training_classes)
for input_features, prediction, actual in zip(testing_inputs[:10],
                                             random_forest_classifier.predict(testing_inputs[:10]),
                                             testing_classes[:10]):
    print('{}\t->\t{}\t(Actual:{})'.format(input_features, prediction, actual))
