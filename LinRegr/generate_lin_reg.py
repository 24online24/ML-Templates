import numpy
import pandas
import pickle
import sklearn
from sklearn import linear_model

data = pandas.read_csv('student-mat.csv', sep=';')

# data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
data = data[['G2', 'G3']]

predict = 'G3'

x = numpy.array(data.drop(columns=predict))
y = numpy.array(data[predict])

best_accuracy = 0
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.1)  # four numpy arrays that contain the training input data and result, respectively the testing input and results

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)

    if accuracy > best_accuracy:
        print('Accuracy: ', accuracy)
        print('Coeficient: ', linear.coef_)
        print('Intercept: ', linear.intercept_)
        print()
        best_accuracy = accuracy
        with open('model2.pickle', 'wb') as output_file:
            pickle.dump(linear, output_file)

# pickle_in = open('model.pickle', 'r')
# linear = pickle.load(pickle_in)

# predictions = linear.predict(x_test)

# for i, prediction in enumerate(predictions):
#     print('Prediction:', prediction)
#     print('Data:', x_test[i])
#     print('Result:', y_test[i])
#     print()

# print('Accuracy: ', accuracy)
# print('Coeficient: ', linear.coef_)
# print('Intercept: ', linear.intercept_)

# print(linear.predict([[20, 20, 0, 0, 0]]))
