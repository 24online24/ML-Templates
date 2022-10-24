import numpy
from sklearn.preprocessing import scale  # Pentru a uniformiza valorile date
from sklearn.datasets import load_digits  # Pentru setul de date
from sklearn.cluster import KMeans  # Pentru modelul de învățare
from sklearn import metrics  # Pentru a da un scor modelului

# Citirea setului de date, în cazul acesta: cifre
digits = load_digits()

# Valorile sunt mari (RGB sau grayscale), iar pentru calcul mai ușor
# sunt reduse uniform la valori între -1 și 1.
data = scale(digits.data)
# y conține "etichetele" grupelor în care clasificăm elementele
y = digits.target

# Pentru a nu trebui introdus numărul de clase manual
k = len(numpy.unique(y))  # k = 10 pentru că sunt 10 cifre

# Valorile pentru numărul de elemente, respectiv numărul de
# caracteristici/ atribute a fiecărui element.
samples, features = data.shape

# Funcție din sklean pentru a afișa diferite "scoruri" pentru modelul antrenat.
# Distanța pentru a calcula apartenența unui punct la un centroid este distanța
# euclidiană.
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (name, estimator.inertia_,
                                                            metrics.homogeneity_score(
                                                                y, estimator.labels_),
                                                            metrics.completeness_score(
                                                                y, estimator.labels_),
                                                            metrics.v_measure_score(
                                                                y, estimator.labels_),
                                                            metrics.adjusted_rand_score(
                                                                y, estimator.labels_),
                                                            metrics.adjusted_mutual_info_score(
                                                                y, estimator.labels_),
                                                            metrics.silhouette_score(data, estimator.labels_, metric='euclidean')))


# Numărul de clustere este k, adică 10 (pentru 10 cifre). Inițializarea centroidelor
# este aleatorie ("random"). "n_init" reprezintă de câte ori va rula algoritmul,
# cu valori inițiale diferite pentru centroide.
classifier = KMeans(n_clusters=k, init='random', n_init=10)

# Afișează scorurile date de funcția mai sus definită.
bench_k_means(classifier, 'Best K Means:', data)
