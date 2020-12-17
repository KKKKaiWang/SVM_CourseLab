import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

# Load dataset
def load_data(fname):
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


# Calculate classification accuracy
def eval_acc(label, pred):
    return np.sum(label == pred) / len(pred)
    

# Visualization
def show_data(data):
    fig, ax = plt.subplots()
    cls = data[:, 2]
    ax.scatter(data[:, 0][cls==1], data[:, 1][cls==1])
    ax.scatter(data[:, 0][cls==-1], data[:, 1][cls==-1])
    ax.grid(False)
    fig.tight_layout()
    plt.show()



class SVM():

    def __init__(self):
        # Todo: initialize SVM class 
        w = np.zeros(200)
        b = 0

    def train(self, data_train):
        # Todo: train model
        X = data_train[:,:2]
        y = data_train[:,2]
        print(X.shape,y.shape)
        y = y.reshape(-1,1) 
        H = np.multiply(np.dot(X,X.T),np.dot(y,y.T))
        m,n = X.shape
        H = np.dot(X,X.T) * np.dot(y,y.T)
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(-np.eye(m))
        h = cvxopt_matrix(np.zeros(m))
        A = cvxopt_matrix(y.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        w = ((y * alphas).T @ X).reshape(-1,1)
        S = (alphas > 1e-4).flatten()
        b = y[S] - np.dot(X[S], w)
        self.w = w.flatten()
        self.b = b[0]
        print('b = ', b[0])
    def predict(self, x):
        # Todo: predict labels
        pred = np.sign(np.dot(x,self.w.reshape(-1,1))+self.b)
        pred = pred.flatten()
        return pred


if __name__ == '__main__':
    # Load dataset
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # dataset format [x1, x2, t], shape (N * 3)
    data_test = load_data(test_file)

    # train SVM
    svm = SVM()
    svm.train(data_train)

    # predict
    x_train = data_train[:, :2]  # features [x1, x2]
    t_train = data_train[:, 2]  # ground truth labels
    t_train_pred = svm.predict(x_train)  # predicted labels
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # evaluate
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))

