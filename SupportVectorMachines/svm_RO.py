import sklearn
from sklearn import datasets  # Pentru setul de date
from sklearn import svm  # Pentru algoritmul de Support Vector Machine
from sklearn import metrics  # Pentru a verifica rezultatul

dataset = datasets.load_breast_cancer()

# Am împărțit datele în grupuri pentru antrenament
# și test în proporții de 80-20
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    dataset.data, dataset.target, test_size=0.2)

# Am definit clasele în care pot fi clasificate elementele
classes = ['malignant', 'benign']

# Am ales funcția kernel drep liniară, iar argumentul "C" reprezintă
# marja de eroare. C=0 reprezintă o limită strictă, în timp ce o
# valoare mai mare a C reprezintă o limită mai permisivă.
clf = svm.SVC(kernel='linear', C=2)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

# Acuratețea modelului antrenat este calulată și afișată
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)
