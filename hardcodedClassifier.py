class HcClassifier:

    def __init__(self):
        self.dataList = ""

    """
    Fit: (alias for train)
    """
    def fit(self, dataList, targetList):
        self.train(dataList, targetList)

    """
    Train:
        Inputs: dataList - the data
                targetList - the corresponding targets for the data columns
        Return: None
    """
    def train(self, dataList, targetList):
        self.dataList = dataList

    """
    Predict:
        Inputs: instanceList - the data columns for which targets will be predicted
        Return: predictionList - the predicted targets for the data in instanceList
    """
    def predict(self, instanceList):
        predictionList = [0] * len(instanceList)
        return predictionList