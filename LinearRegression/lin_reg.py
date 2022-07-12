from matplotlib import style
import matplotlib.pyplot
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

pickle_in = open('model2.pickle', 'rb')
linear = pickle.load(pickle_in)

# predictions = linear.predict(x)

# for i, prediction in enumerate(predictions):
#     print('Prediction:', prediction)
#     print('Data:', x[i])
#     print('Result:', y[i])
#     print()

accuracy = linear.score(x, y)

print('Accuracy: ', accuracy)
print('Coeficient: ', linear.coef_)
print('Intercept: ', linear.intercept_)

# print(linear.predict([[20, 20, 0, 0, 0]]))

p = 'G2'

style.use('ggplot')
matplotlib.pyplot.scatter(data[p], data['G3'])
matplotlib.pyplot.xlabel(p)
matplotlib.pyplot.ylabel('Final grade')
matplotlib.pyplot.show()
# print(linear.predict([[20, 20, 56, 0, 0]]))