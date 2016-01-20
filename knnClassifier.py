"""
k-Nearest Neighbors Classifier
Requires the data to be scaled
"""
class KnnClassifier:

    def __init__(self, k):
        self.k = k

    """
    Fit: (alias for train)
    """
    def fit(self, dataList, targetList):
        self.train(dataList, targetList)

    """
    Train:
        Inputs: dataList - the data columns
                targetList - the corresponding targets for the data columns
        Return: None
        Description:
            Training for kNN simply means saving the data.
    """
    def train(self, dataList, targetList):
        self.dataList = dataList
        self.targetList = targetList

    """
    Predict:
        Inputs: instanceList - the data columns for which targets will be predicted
        Return: predictionList - the predicted targets for the data in instanceList
        Description:
            Uses the Euclidean distance to determine the nearest neighboring point.
            The class of the nearest point is the prediction.
    """
    def predict(self, instanceList):
        #initialize the lists
        predictionList = [0] * len(instanceList)

        print ("Length of instanceList is " + str(len(instanceList)))
        for i in range(len(instanceList)):
            # get a list of the classes of the nearest neighbors
            point = instanceList[i]
            predictionList[i] = self.getNearestNeighbor(point)
            if (i % 100 == 0 and i != 0):
                print "Processed %s/%s..." % (str(i), str(len(instanceList)))

        print "Completed: %s/%s..." % (str(len(instanceList)), str(len(instanceList)))
        return predictionList

    """
    GetNearestNeighbor()
        Input: a point in the dimension space
        Return: list of the classes of the k-nearest neighbors
        Description:
            Calculates the Euclidean distance between the index point and all other points. Uses these distances
            to determine the k-nearest neighbors
    """
    def getNearestNeighbor(self, pointA):
        # this assumes that each row in both the training data and instances for which we are predicting
        # have the same number of attributes
        numAttributes = len(pointA)

        # list containing the distance from pointA and the respective index for self.dataList
        distanceList = []
        neighborClasses = []

        # calculate each distance
        for i in range(len(self.dataList)):
            distance = self.calcDistance(pointA, self.dataList[i])
            distanceList.append([distance, self.targetList[i]])

        # determine classes of the the k-nearest neighbors
        sortedDistanceList = sorted(distanceList, key=lambda distanceList: distanceList[0])
        for i in range(self.k):
            neighborClasses.append(sortedDistanceList[i][1])

        # find mode class of nearest neighbors
        guess = max(set(neighborClasses), key=neighborClasses.count)
        return guess

    """ Calculate the Euclidean distance between two points (NOTE: does not compute square root) """
    def calcDistance(self, pointA, pointB):
        sum = 0
        for i in range(len(pointA)):
            sum = sum + (pointA[i] - pointB[i]) ** 2
        return sum




