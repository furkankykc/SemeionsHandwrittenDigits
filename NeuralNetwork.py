import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection  import KFold
import matplotlib.pyplot as plt
# import pickle


# import matplotlib.pyplot as plt

class Network:

    def __init__(self,
                 num_nodes_in_layers,
                 batch_size,
                 num_epochs,
                 learning_rate,
                 weights_file
                 ):

        self.num_nodes_in_layers = num_nodes_in_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weights_file = weights_file

        # build the network
        #         w1/b1    w2/b2
        # 784(inputs) ---> 20 ---> 10(output)
        #         x     z1  a1  z2  a2=y
        self.weight1 = np.random.normal(0, 1, [self.num_nodes_in_layers[0], self.num_nodes_in_layers[1]])
        self.bias1 = np.zeros((1, self.num_nodes_in_layers[1]))
        self.weight2 = np.random.normal(0, 1, [self.num_nodes_in_layers[1], self.num_nodes_in_layers[2]])
        self.bias2 = np.zeros((1, self.num_nodes_in_layers[2]))
        self.loss = []

    def train(self, inputs, labels):

        for epoch in range(self.num_epochs):  # training begin
            iteration = 0
            while iteration < len(inputs):
                # batch input
                inputs_batch = inputs[iteration:iteration + self.batch_size]
                labels_batch = labels[iteration:iteration + self.batch_size]

                # forward pass
                z1 = np.dot(inputs_batch, self.weight1) + self.bias1
                a1 = self.relu(z1)
                z2 = np.dot(a1, self.weight2) + self.bias2
                y = self.softmax(z2)

                # calculate loss
                loss = self.cross_entropy(y, labels_batch)
                loss += self.L2_regularization(0.01, self.weight1, self.weight2)  # lambda
                self.loss.append(loss)

                # backward pass
                delta_y = (y - labels_batch) / y.shape[0]
                delta_hidden_layer = np.dot(delta_y, self.weight2.T)
                # print "y shape = ",y.shape
                # print delta_y.shape
                # print self.weight2.shape
                # print delta_hidden_layer.shape
                # print a1.shape
                delta_hidden_layer[a1 <= 0] = 0  # derivatives of relu

                # backpropagation
                weight2_gradient = np.dot(a1.T, delta_y)  # forward * backward
                bias2_gradient = np.sum(delta_y, axis=0, keepdims=True)

                weight1_gradient = np.dot(inputs_batch.T, delta_hidden_layer)
                bias1_gradient = np.sum(delta_hidden_layer, axis=0, keepdims=True)

                # L2 regularization
                weight2_gradient += 0.01 * self.weight2
                weight1_gradient += 0.01 * self.weight1

                # stochastic gradient descent
                self.weight1 -= self.learning_rate * weight1_gradient  # update weight and bias
                self.bias1 -= self.learning_rate * bias1_gradient
                self.weight2 -= self.learning_rate * weight2_gradient
                self.bias2 -= self.learning_rate * bias2_gradient

                #print('=== Epoch: {:d}/{:d}\tIteration:{:d}\tLoss: {:.2f} ===').format(epoch + 1, self.num_epochs,iteration + 1, loss)
                iteration += self.batch_size

            obj = [self.weight1, self.bias1, self.weight2, self.bias2]
            with open('weights.pkl', 'wb') as handle:
                pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def test(self, inputs, labels):
        input_layer = np.dot(inputs, self.weight1)
        hidden_layer = self.relu(input_layer + self.bias1)
        scores = np.dot(hidden_layer, self.weight2) + self.bias2
        probs = self.softmax(scores)
        acc = float(np.sum(np.argmax(probs, 1) == labels)) / float(len(labels))
        print('Test accuracy: {:.2f}%').format(acc * 100)

        network.drawLoss(str(acc*100))
        return acc*100

    def predict(self,inputs):
        input_layer = np.dot(inputs, self.weight1)
        hidden_layer = self.relu(input_layer + self.bias1)
        scores = np.dot(hidden_layer, self.weight2) + self.bias2
        return scores
    # activation function
    def relu(self,inputs):
        return np.maximum(inputs, 0)

    # output probability distribution function
    def softmax(self,inputs):
        exp = np.exp(inputs)
        return exp / np.sum(exp, axis=1, keepdims=True)

    # loss
    def cross_entropy(self,inputs, y):
        indices = np.argmax(y, axis=1).astype(int)
        probability = inputs[np.arange(len(inputs)), indices]  # inputs[0, indices]
        log = np.log(probability)
        loss = -1.0 * np.sum(log) / len(log)
        return loss

    # L2 regularization
    def L2_regularization(self,la, weight1, weight2):
        weight1_loss = 0.5 * la * np.sum(weight1 * weight1)
        weight2_loss = 0.5 * la * np.sum(weight2 * weight2)
        return weight1_loss + weight2_loss

    def drawLoss(self,acc="Unknown"):
        plt.text(0, 30, 'Test accuracy :{:s} '.format(acc), style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

        plt.plot(self.loss)
        plt.suptitle('hidden_layers:{:d} \n batch_size:{:d} \n epochs:{:d} \n learning_rate:{:f}'.format(self.num_nodes_in_layers[1],self.batch_size,self.num_epochs,self.learning_rate), fontsize=8, fontweight='bold')
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.show()
    def info(self):
        print self.num_nodes_in_layers[1],self.batch_size,self.num_epochs,self.learning_rate

    def kfoldTestScore(self,X, y, split=4):
        MSE = 0
        meanErr = 0
        perf = 0
        kfold = KFold(n_splits=split, shuffle=True, random_state=0)



        for ind_train, ind_test in kfold.split(X):
            # print ind_test,ind_train
            dataTest = X[ind_test]
            dataTrain = X[ind_train]
            targetTest = y[ind_test]
            targetTrain = y[ind_train]

            self.train(dataTrain, targetTrain)
            meanErr+= np.sum(self.loss)/len(self.loss)
            MSE += (np.mean(np.mean(np.square(targetTest - self.predict(dataTest)))))  # mean sum squared loss
            perf+=self.test(dataTest,(np.argmax(targetTest,1)))

        print "MSE:",MSE / split
        print "Mean Error:",meanErr / split
        print "10-Fold Accuracy:", perf / split
        print " {:d} {:d} {:d} {:.5f} {:.2f} {:.2f} {:.2f}%".format(self.num_nodes_in_layers[1],self.batch_size,self.num_epochs,self.learning_rate,MSE / split,meanErr / split,perf / split)


network =  Network(
                 num_nodes_in_layers = [256, 20, 10],
                 batch_size = 10,
                 num_epochs = 1000,
                 learning_rate = 0.001,
                 weights_file = 'weights.pkl'
             )


