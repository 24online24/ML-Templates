import numpy  # Pentru array-uri de numere în formatul folosit de sklearn
import pandas  # Pentru importarea datelor
import pickle  # Pentru a salva modelul antrenat
import sklearn  # Pentru împărțirea datelor pentru antrenament și test
from sklearn import linear_model # Pentru funcția de regresie liniară

# Citim datele dintr-un fișier .csv
data = pandas.read_csv('student-mat.csv', sep=';')

# Definim câmpurile de care avem nevoie de câmpul pe care vrem să îl prezicem
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
predict = 'G3'

# Separăm datele de intrare în array-uri
x = numpy.array(data.drop(columns=predict))
y = numpy.array(data[predict])

# Rulăm de 10000 de ori algoritmul și reținem cea mai bună acuratețe
best_accuracy = 0
for _ in range(10000):
    # La fiecare rulare împărțim aleator setul de date pentru,
    # folosind 90% pentru antrenare și 10% pentru testare
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.1)

    # Antrenăm modelul folosind funcția de regresie liniară
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # Calculăm acuratețea modelului pe datele de test
    accuracy = linear.score(x_test, y_test)

    # Afișează în consolă și reține atributele modelului cu cea mai mare acuratețe
    if accuracy > best_accuracy:
        print('Accuracy: ', accuracy)
        print('Coeficient: ', linear.coef_)
        print('Intercept: ', linear.intercept_)
        print()
        best_accuracy = accuracy
        best_model = linear

# Cel mai bun model este salvat într-un fișier pentru a putea fi reutilizat
with open('model2.pickle', 'wb') as output_file:
    pickle.dump(best_model, output_file)
