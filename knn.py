from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy

class knn:

    def __init__(self):
        df = pd.read_csv('completeData.csv', sep=',')
        self.model = KNeighborsClassifier(n_neighbors=1)

        classifiers = numpy.array(df[df.columns[2]])
        vectors = numpy.array(df[df.columns[3:100]])

        self.model.fit(vectors, classifiers)

    def predict(self, vector):
        return self.model.predict(vector)