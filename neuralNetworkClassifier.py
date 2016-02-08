"""
Neural Network Classifier
Requires the data to be scaled
"""
import random
"""
Neuron Class - used by the neural network
"""
class Neuron:
    def __init__(self, numInputs, threshold):
        self.threshold = threshold

        # initialize the weights to small random numbers
        self.weights = [];
        for i in range(numInputs):
            self.weights.append(random.uniform(-1.0,1.0))

    """ Adjust the weight of an input """
    def adjustWeight(self, weightIndex, learningRate, target, input):
        weights[i] = weights[i] - learningRate * (self.output) * input

    def getWeight(self, index):
        return self.weights[i]

    def setWeight(self, index, newWeight):
        self.weights[i] = newWeight

    """ Calculate the output of the neuron """
    def calcOutput(self, inputs):
        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]
        if sum <= self.threshold:
            return 0
        else:
            return 1

"""
Neural Network Classifier
"""
class NeuralNetworkClassifier:

    def __init__(self, numNeurons, numInputs, threshold):
        numInputs += 1  # account for the bias input
        self.numNeurons = numNeurons
        self.neuronList = []

        # Construct the network
        for i in range(numNeurons):
            n = Neuron(numInputs, threshold)
            self.neuronList.append(n)

        self.hasBeenTrained = False
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
        return

    """
    Predict:
        Inputs: instanceList - the data columns for which targets will be predicted
        Return: predictionList - the predicted targets for the data in instanceList
        Description:
            Yis.
    """
    def predict(self, instanceList):
        instanceList = instanceList.tolist()
        outputList = []
        for i in range(len(instanceList)):
            neuronOutputs = []
            inputs = instanceList[i]
            inputs.insert(0,-1)    # insert the bias
            for j in range(len(self.neuronList)):
                neuronOutputs.append(self.neuronList[j].calcOutput(inputs))
            outputList.append(neuronOutputs)
        print(outputList)

        predictionList = []
        return predictionList