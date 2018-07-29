import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from NeuralNetwork import network


def main():
    df = pd.read_csv("data/semeion.data", delimiter=r"\s+",
                     header=None)

    X = pd.DataFrame(df)
    X = X.drop([256, 257, 258, 259, 260, 261, 262, 263, 264, 265], axis=1)
    label_df = pd.DataFrame(df.iloc[:, [256, 257, 258, 259, 260, 261, 262, 263, 264, 265]])
    label_df.rename(columns={256: 0, 257: 1, 258: 2, 259: 3, 260: 4, 261: 5, 262: 6, 263: 7, 264: 8, 265: 9},
                    inplace=True)
    y = label_df

    print "yyyy = ", y.shape
    print "x = ", X.shape
    X_train = np.array(X).astype('float32')
    y_train = np.array(y)

    # network.train(x_train, y_train)
    # network.test(X_train,(np.argmax(y_train,1)))
    # network.kfoldTestScore(X_train,y_train,split=10)
    # 1 0 0 0 0 0 0 0 0 0
    y = label_df.apply(lambda x: label_df.columns[x.idxmax()], axis=1)

    predict(X, y)
def predict(test_images, label_images):
    num = 0
    with open('weights.pkl', 'rb') as handle:
        b = pickle.load(handle)

    weight1 = b[0]
    bias1 = b[1]
    weight2 = b[2]
    bias2 = b[3]
    nn = network
    while num < test_images.shape[0]:
        input_layer = np.dot(test_images[num:num + 1], weight1)
        hidden_layer = nn.relu(input_layer + bias1)
        scores = np.dot(hidden_layer, weight2) + bias2
        probs = nn.softmax(scores)
        predict = np.argmax(probs)
        X_img = test_images.iloc[:, :].values.astype('float32')
        X_img = X_img.reshape(-1, 16, 16)

        num += 1

    for i in range(500, 505):
        plt.subplot(330 + (i + 1))
        plt.imshow(X_img[i], cmap=plt.get_cmap('gray'))
        plt.title(label_images[i]);
    plt.show()




if __name__ == "__main__":
    main()