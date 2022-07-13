import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas

data = pandas.read_csv('car.data')

# Transforming non-numerical values to numerical
lab_enc = sklearn.preprocessing.LabelEncoder()
buying = lab_enc.fit_transform(list(data["buying"]))
maint = lab_enc.fit_transform(list(data["maint"]))
doors = lab_enc.fit_transform(list(data["doors"]))
persons = lab_enc.fit_transform(list(data["persons"]))
lug_boot = lab_enc.fit_transform(list(data["lug_boot"]))
safety = lab_enc.fit_transform(list(data["safety"]))
quality_class = lab_enc.fit_transform(list(data["class"]))

# x is the data that will be used to predict y
x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(quality_class)  
# Deviding the data 90% for training and 10% for tests
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1)

best_accuracy = 0
for nn in range(1, 25):
    model = KNeighborsClassifier(n_neighbors=nn)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    # print(nn, accuracy)
    # Best accuracy and number of neighbors get saved
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_nn = nn

print(best_nn, best_accuracy)

# predicted = model.predict(x_test)
# names = ['unacc', 'acc', 'good', 'vgood']

# for i, prediction in enumerate(predicted):
#     print('Predicted: ', names[prediction],
#           'Data: ', x_test[i], 'Actual: ', names[y_test[i]])
#     neighbors = model.kneighbors([x_test[i]], 10, True)
#     print('Neighbors: ', neighbors)
