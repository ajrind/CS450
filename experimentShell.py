import sys
import csv
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from hardcodedClassifier import HcClassifier as hcClass
from knnClassifier import KnnClassifier as knnClass
from neuralNetworkClassifier import NeuralNetworkClassifier as nnClass
from sklearn.neighbors import KNeighborsClassifier

"""The experiment shell"""
def main(argv):
    # Determine which dataset to use
    filename = getUserDataChoice()

    # Read in the dataset
    theData = []
    theTargets = []
    readFile(filename, theData, theTargets)

    # Preprocess the data
    preprocessData(filename, theData, theTargets)

    # Split the dataset into two sets: a training set and a testing set
    testScale = 0.33
    trainData, testData, trainTargets, testTargets = train_test_split(theData, theTargets, test_size=testScale)

    # Create and apply the scaler
    if shouldScaleData():
        scaler = preprocessing.StandardScaler().fit(trainData)
        trainData = scaler.transform(trainData)
        testData  = scaler.transform(testData)
        print("The data has been scaled.")
    else:
        print("The data will not be scaled.")

    print ("Original List: %d", len(testData[0]) - 1)

    # Determine which classifier to use
    userClassChoice = int(getUserClassChoice())
    classifier = None
    # 1. hardcoded classifier
    if userClassChoice == 1:
        classifier = hcClass()
    # 2. my k-Nearest Neighbors
    elif userClassChoice == 2:
        print "How many neighbors should be used? (Test data will contain %d rows)" % len(testData)
        k = int(raw_input())
        if int(k) > len(testData):
            print "Invalid input! Defaulting to k = 3..."
            k = 3
        classifier = knnClass(int(k))
    # 3. sklearn k-Nearest Neighbors
    elif userClassChoice == 3:
        print "How many neighbors should be used? (Test data will contain %d rows)" % len(testData)
        k = int(raw_input())
        if int(k) > len(testData) or int(k) < 1:
            print "Invalid input! Defaulting to k = 3..."
            k = 3
        classifier = KNeighborsClassifier(n_neighbors=k)
    # 4. my Decision Tree (in progress)
    elif userClassChoice == 4:
        print("This classifier is not yet implemented. Exiting...")
        exit()
    # 5. sklearn Decision Tree (in progress)
    elif userClassChoice == 5:
        print("This classifier is not yet implemented. Exiting...")
        exit()
    # 6. my Neural Network
    elif userClassChoice == 6:
        print "How many nodes should each layer contain?"
        numNodes = int(raw_input())
        if int(numNodes) <= 0:
            print "Invalid input! Defaulting to 3 nodes..."
            numNodes = 3
        threshold = 0
        print("! testData[0]=",testData[0])
        classifier = nnClass(numNodes, len(testData[0]), threshold)
    else:
        print("%s is not a valid choice. Exiting...", userClassChoice)
        exit()


    # Train the classifier and make predictions
    classifier.fit(trainData, trainTargets)
    predictionList = classifier.predict(testData)

    # Determine the accuracy of the classifier
    numCorrect = calcNumCorrect(predictionList, testTargets)

    # Display the results
    percent = str(round(float(numCorrect) / len(testData) * 100, 3)) + "%"
    print'Correctly predicted: %d of %d (%s)' % (numCorrect, len(testData), percent)
    return
#end main


"""Gets user input for which dataset to use"""
def getUserDataChoice():
    print("Please enter the number of the data set you wish to use:")
    print("1 = Iris")
    print("2 = Car Evaluation")
    print("3 = Pima Indians Diabetes")
    choice = int(raw_input())

    filename = ""
    if choice == 1:
        filename = "iris.data"
    elif choice == 2:
        filename = "car.data"
    elif choice == 3:
        filename = "pima-indians-diabetes.data"
    else:
        print("%s is not a valid choice. Exiting...", choice)
        exit()

    return filename


"""Gets user input for which classifier to use"""
def getUserClassChoice():
    print("Please enter the number of the data set you wish to use:")
    print("1 = Hardcoded Classifier")
    print("2 = k-Nearest Neighbors")
    print("3 = k-Nearest Neighbors (from sklearn)")
    print("4 = Decision Tree [NOT YET IMPLEMENTED]")
    print("5 = Decision Tree (from sklearn) [NOT YET IMPLEMENTED]")
    print("6 = Neural Network")
    choice = raw_input()
    return choice


def readFile(filename, theData, theTargets):
    # File layout: last column contains the targets, all other columns are the data
    with open(filename) as csvFile:
        # read in the CSV
        reader = csv.reader(csvFile)
        for row in reader:
            theData.append(row[0:(len(row) - 1)])
            theTargets.append(row[len(row) - 1])
    return

"""Prompts whether or not the user desires to preprocess the data (apply a scalar)"""
def shouldScaleData():
    print("Scale the Data? (y/n)")
    input = raw_input()
    if input == 'y':
        return True
    else:
        return False

"""Returns a list of all the possible targets in the data"""
def getPossibleTargets(filename, theData, theTargets):
    # Create the list of all different targets
    possibleTargets = []
    for t in theTargets:
        if t not in possibleTargets:
            possibleTargets.append(t)
    return possibleTargets


"""Converts strings to numbers in the data"""
def preprocessData(filename, theData, theTargets):
    possibleTargets = getPossibleTargets(filename, theData, theTargets)

    # Convert the targets into a numeric form
    for i in range(len(theTargets)):
        theTargets[i] = possibleTargets.index(theTargets[i])

    # Special preprocessing for the car data set: convert relative values to integers
    if filename == "car.data":
        # Column 1: vhigh, high, med, low.
        for i in range(len(theData)):
            if theData[i][0] == 'low':
                theData[i][0] = 1
            if theData[i][0] == 'med':
                theData[i][0] = 2
            if theData[i][0] == 'high':
                theData[i][0] = 3
            if theData[i][0] == 'vhigh':
                theData[i][0] = 4

        # Column 2: vhigh, high, med, low.
        for i in range(len(theData)):
            if theData[i][1] == 'low':
                theData[i][1] = 1
            if theData[i][1] == 'med':
                theData[i][1] = 2
            if theData[i][1] == 'high':
                theData[i][1] = 3
            if theData[i][1] == 'vhigh':
                theData[i][1] = 4

        # Column 3: 2, 3, 4, 5more.
        for i in range(len(theData)):
            if theData[i][2] == '2':
                theData[i][2] = 1
            if theData[i][2] == '3':
                theData[i][2] = 2
            if theData[i][2] == '4':
                theData[i][2] = 3
            if theData[i][2] == '5more':
                theData[i][2] = 4
        # Column 4: 2, 4, more.
        for i in range(len(theData)):
            if theData[i][3] == '2':
                theData[i][3] = 1
            if theData[i][3] == '4':
                theData[i][3] = 2
            if theData[i][3] == 'more':
                theData[i][3] = 3

        # Column 5: small, med, big.
        for i in range(len(theData)):
            if theData[i][4] == 'small':
                theData[i][4] = 1
            if theData[i][4] == 'med':
                theData[i][4] = 2
            if theData[i][4] == 'big':
                theData[i][4] = 3

        # Column 6: low, med, high.
        for i in range(len(theData)):
            if theData[i][5] == 'low':
                theData[i][5] = 1
            if theData[i][5] == 'med':
                theData[i][5] = 2
            if theData[i][5] == 'high':
                theData[i][5] = 3
    return


"""Calculate the number of correct predictions"""
def calcNumCorrect(predictionList, testTargets):
    count = 0
    if len(predictionList) != len(testTargets):
        print("calcNumCorrect: len(predictionList) = %d and len(testTargets) = %d") % (len(predictionList), len(testTargets))
        print("calcNumCorrect: sizes of the lists do not match! Exiting...")
        exit()
    size = len(testTargets)
    for i in range(size):
        if predictionList[i] == testTargets[i]:
            count += 1
    return count


# establish main
if __name__ == "__main__":
    main(sys.argv)
