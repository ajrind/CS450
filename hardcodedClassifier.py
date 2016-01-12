class HcClassifier:

    def __init__(self):
        self.dataSet = ""

    def train(self, dataSet, targetSet):
        self.dataSet = dataSet

    def predict(self, instanceSet):
        predictions = [0] * len(instanceSet)
        return predictions