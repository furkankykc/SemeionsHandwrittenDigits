from NeuralNetwork import network
import pandas as pd
import numpy as np
def test():
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
    nn = network
    nn.kfoldTestScore(X_train,y_train)




if __name__ == "__main__":
    test()