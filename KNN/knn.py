import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas

data = pandas.read_csv('car.data')
# print(data.head())

label_encoder = sklearn.preprocessing.LabelEncoder()
buying = label_encoder.fit_transform(list(data["buying"]))
maint = label_encoder.fit_transform(list(data["maint"]))
doors = label_encoder.fit_transform(list(data["doors"]))
persons = label_encoder.fit_transform(list(data["persons"]))
lug_boot = label_encoder.fit_transform(list(data["lug_boot"]))
safety = label_encoder.fit_transform(list(data["safety"]))
quality_class = label_encoder.fit_transform(list(data["class"]))

predict = 'quality_class'

x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(quality_class)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1)

model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)

predicted = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood']

for i, prediction in enumerate(predicted):
    print('Predicted: ', names[prediction],
          'Data: ', x_test[i], 'Actual: ', names[y_test[i]])
    neighbors = model.kneighbors([x_test[i]], 10, True)
    print('Neighbors: ', neighbors)
