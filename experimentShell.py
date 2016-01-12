import time
import sys
import numpy as np
from hardcodedClassifier import HcClassifier

def main(argv):
    # Import and setup the Iris dataset
    from sklearn import datasets 
    iris = datasets.load_iris() 
    irisData = iris.data
    irisTargets = iris.target


    # Shuffle the dataset
    seed = int(time.time())
    np.random.seed(seed)
    np.random.shuffle(irisData)
    np.random.seed(seed)            # reseed to make targets shuffle in the same order as data
    np.random.shuffle(irisTargets)


    # Split the dataset into two sets: a training set (70%) and a testing set (30%)
    trainSize = int(.7 * irisTargets.size)
    testSize = irisTargets.size - trainSize

    trainData = irisData[:trainSize]
    testData = irisData[trainSize:trainSize + testSize]

    trainTargets = irisTargets[:trainSize]
    testTargets = irisTargets[trainSize:trainSize + testSize]


    # Train the classifier and make predictions
    hcc = HcClassifier()
    hcc.train(trainData, trainTargets)
    predictionList = hcc.predict(testData)


    # Determine the accuracy of the classifier
    numCorrect = 0
    for i in range(testSize):
        if predictionList[i] == testTargets[i]:
            numCorrect += 1

    output = "Correctly predicted: " + str(numCorrect) + " of " + str(testSize) + " (" + str(round(float(numCorrect) / testSize * 100, 3)) + "%)."
    print output

    return

# establish main    
if __name__ == "__main__":
    main(sys.argv)
