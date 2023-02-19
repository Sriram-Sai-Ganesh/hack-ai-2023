import xgboost as xgb
import numpy as np

class xgbModel:

    def __init__(self):
        self.classifier = xgb.XGBClassifier()
        self.classifier.load_model('basicModel.json')

    # Returns [0,1,2,3] for [thumbs up, thumbs down, open, closed]
    def classify(self, vector):
        data = self.transformData(vector)
        return self.classifier.predict(data)

    def transformData(self,coordinateList):
        x9 = coordinateList[9][1]
        y9 = coordinateList[9][2]

        # Normalize the values
        coord_list = []
        for subList in coordinateList:
            coord_list.append(subList[1] - x9)
            coord_list.append(subList[2] - y9)

        vector = np.array(coord_list)
        vector = vector.reshape(1,42)
        return vector
