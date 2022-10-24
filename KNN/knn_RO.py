import sklearn  # Pentru funcțiile ce prelucrează datele înainte de antrenarea modelului
from sklearn.neighbors import KNeighborsClassifier # Pentru alforitmul de clasificare knn
import pandas # Pentru importarea datelor din fișierul .csv

data = pandas.read_csv('car.data')

# Transformarea valorilor nenumerice în valori numerice
lab_enc = sklearn.preprocessing.LabelEncoder()
buying = lab_enc.fit_transform(list(data["buying"]))
maint = lab_enc.fit_transform(list(data["maint"]))
doors = lab_enc.fit_transform(list(data["doors"]))
persons = lab_enc.fit_transform(list(data["persons"]))
lug_boot = lab_enc.fit_transform(list(data["lug_boot"]))
safety = lab_enc.fit_transform(list(data["safety"]))
quality_class = lab_enc.fit_transform(list(data["class"]))

# Separarea datelor în date de intrare (x) și de ieșire (y)
x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(quality_class)

# 90% din date vor fi folosite pentru antrenament și 10% pentru teste
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1)

# Se încearcă antrenarea modelului pentru un număr variat de vecini,
# începând cu 1 și ajungând la 25. Cea mai bună acuratețe este salvată
best_accuracy = 0
for nn in range(1, 25):
    model = KNeighborsClassifier(n_neighbors=nn)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    # Afișăm acuratețea pentru fiecare număr de vecini
    print(nn, accuracy)
    # Cea mai bună acuratețe și numărul de vecini sunt salvate
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_nn = nn
# Se afișează rezultatul cel mai bun
print(best_nn, best_accuracy)

# Folosind modelul memorat prezicem calitatea celoralalte mașini,
# din segmentul lăsat pentru testare
predicted = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood']

for i, prediction in enumerate(predicted):
    # print('Predicted: ', names[prediction],
    #       'Data: ', x_test[i], 'Actual: ', names[y_test[i]])
    neighbors = model.kneighbors([x_test[i]], best_nn, True)
    print('Neighbors: ', neighbors)
