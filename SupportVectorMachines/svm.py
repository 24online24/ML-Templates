import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

dataset = datasets.load_breast_cancer()

# print(dataset.feature_names)
# print(dataset.target_names)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    dataset.data, dataset.target, test_size=0.2)

# print(x_train)
# print(y_train)
classes = ['malignant', 'benign']

clf = svm.SVC(kernel='linear', C=2)

# clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)
