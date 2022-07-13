import numpy
import pandas
import pickle
import sklearn
from sklearn import linear_model

data = pandas.read_csv('student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
predict = 'G3'

x = numpy.array(data.drop(columns=predict))
y = numpy.array(data[predict])

# Different data is used for training in every run. This results in different model accuracy.
best_accuracy = 0
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train) 
    accuracy = linear.score(x_test, y_test)

    # Prints to console the attributes of the new most accurate model
    if accuracy > best_accuracy:
        print('Accuracy: ', accuracy)
        print('Coeficient: ', linear.coef_)
        print('Intercept: ', linear.intercept_)
        print()
        best_accuracy = accuracy
        best_model = linear

# Tthe best model is written to the output
with open('model2.pickle', 'wb') as output_file:
    pickle.dump(best_model, output_file)
