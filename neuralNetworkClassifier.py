"""
Neural Network Classifier
Requires the data to be scaled
"""
import random
"""
Neuron Class - used by the neural network
    Inputs: numInputs - One for each attribute and one for the bias
"""
class Neuron:
    e = 2.7182818284
    bias = -1
    def __init__(self, numInputs):
        self.equasion = ""
        # initialize the weights to small random numbers
        self.inputWeights = []
        self.output = None
        numInputs += 1 # account for the bias (the last weight in the list is for the bias)
        for i in range(numInputs):
            self.inputWeights.append(random.uniform(-1.0,1.0))

    """ Adjust the weight of an input """
    def adjustWeight(self, weightIndex, learningRate, target, input):
       self.inputWeights[i] = self.inputWeights[i] - learningRate * (self.output) * input

    def setWeight(self, index, newWeight):
        self.inputWeights[i] = newWeight

    """ Calculate the output of the neuron: using the sigmoid function """
    def calcOutput(self, inputs):
        equasion = ""
        sum = 0
        # All inputs (excluding the bias)
        for i in range(len(inputs)):
            sum += (inputs[i] * self.inputWeights[i])
            equasion += (str(inputs[i]) + " * " + str(self.inputWeights[i]) + " + ")

        # The bias input
        sum += (Neuron.bias * self.inputWeights[-1])
        equasion += ("-1 * " + str(self.inputWeights[-1]))

        # Sigmoid function is the activation function
        self.output = 1/(1 + Neuron.e**(sum * (-1)))

        #print("equasion: " + equasion)
        #print("sum: %s" % sum)
        #print("output: %s" % self.output)
        return

    """ Returns the output of this neuron """
    def getOutput(self):
        return self.output

    def display(self):
        print("      Weights: ", self.inputWeights)
        print("      Output:  ", self.output)
"""
Neural Network Classifier
    Inputs:
"""
class NeuralNetworkClassifier:
    numEpochs = 1

    def __init__(self, numLayers, listNumNeuronsPerLayer, numAttributes):
        self.numLayers = numLayers
        self.listNumNeuronsPerLayer = listNumNeuronsPerLayer
        self.networkLayers = []

        # construct the network
        for i in range(numLayers):
            neuronLayer = []

            # determine number of inputs to each node in the current layer
            if i == 0:
                numInputs = numAttributes
            else:
                numInputs = len(self.networkLayers[i - 1])

            # construct the layer of neurons
            for j in range(listNumNeuronsPerLayer[i]):
                neuronLayer.append(Neuron(numInputs))
            self.networkLayers.append(neuronLayer)

        return

    """
    Fit: (alias for train)
    """
    def fit(self, dataList, targetList):
        self.train(dataList, targetList)
        return

    """
    Train:
        Inputs: dataList - the data columns
                targetList - the corresponding targets for the data columns
        Return: None
        Description:
            Training for a neural network involves adjusting weights...
    """
    def train(self, dataList, targetList):
        #for each epoch
        for i in range(NeuralNetworkClassifier.numEpochs): # i is the number of the epoch
            #feed each data point into the
            for j in range(len(dataList)):                # j is the number of the instance in the dataList
                # calculate the output of each layer and feed it through the network (feed forward)
                for k in range(len(self.networkLayers)):   # k is the number of the neuron layer in the network
                    # determine the inputs to use
                    inputList = []
                    if k == 0: # inputs come from the data set
                        inputList = dataList[j]
                    else: # inputs come from the previous layer in the network
                        inputList = self.getListOutputsFromLayer(k - 1)
                    # feed forward
                    self.feedInputsIntoLayer(inputList, k)

                    # error propogation / weight changing

        #self.displayNetwork(dataList[0]) # remove this: debug code
        return

    def getListOutputsFromLayer(self, layerIndex):
        outputList = []
        for i in range(len(self.networkLayers[layerIndex])):
           outputList.append(self.networkLayers[layerIndex][i].getOutput())
        return outputList

    def feedInputsIntoLayer(self, inputList, layerIndex):
        for i in range(len(self.networkLayers[layerIndex])):
            self.networkLayers[layerIndex][i].calcOutput(inputList)
        return

    def feedInputsIntoNetwork(self, inputs):
        for i in range(len(self.networkLayers)):   # i is the number of the neuron layer in the network
            # determine the inputs to use
            inputList = []
            if i == 0: # inputs come from the data set
                inputList = inputs
            else: # inputs come from the previous layer in the network
                inputList = self.getListOutputsFromLayer(i - 1)
            # feed forward
            self.feedInputsIntoLayer(inputList, i)


    def displayNetwork(self, inputs):
        print("**********Displaying the current network state************")
        print"inputs are: %s" % inputs
        for i in range(len(self.networkLayers)): # layers
            print("Layer %d:") % i
            for j in range(len(self.networkLayers[i])):
                print("   Neuron %d:") % j
                self.networkLayers[i][j].display()

    """
    Predict:
        Inputs: instanceList - the data columns for which targets will be predicted
        Return: predictionList - the predicted targets for the data in instanceList
    """
    def predict(self, instanceList):
        instanceList = instanceList.tolist()
        predictionList = []

        for i in range(len(instanceList)):
            self.feedInputsIntoNetwork(instanceList[i])
            prediction = self.getPredictedClass()
            predictionList.append(prediction)

        return predictionList

    def getPredictedClass(self):
        highestOutput = 0
        predictedClass = -1
        for i in range(len(self.networkLayers[-1])): # the target is the node in the output layer with the highest output
            if self.networkLayers[-1][i].getOutput() > highestOutput:
                highestOutput = self.networkLayers[-1][i].getOutput()
                predictedClass = i
        return predictedClass
    """
    Plot Learning Graph:
        Output: a graph of the accuracy of the algorithm after each epoch
    """
    def plotLearningGraph(self):
        stuff = None



    """ Notes:
        activation function is the sigmoid function: 1/(1 + e^(-x))
    """