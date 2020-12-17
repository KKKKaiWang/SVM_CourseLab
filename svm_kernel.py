import numpy as np
import matplotlib.pyplot as plt
import math
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
    ax.scatter(data[:, 0][cls == 1], data[:, 1][cls == 1])
    ax.scatter(data[:, 0][cls == -1], data[:, 1][cls == -1])
    ax.grid(False)
    fig.tight_layout()
    plt.show()


def kernel(i, j, l=1):
    # define kernel function
    return math.exp(-l*np.linalg.norm(i-j))


class SVM():

    def __init__(self):
        # Todo: initialize SVM class
        self.alpha = np.zeros(200)
        self.E = np.zeros(200)
        self.b = 0
        self.X = np.zeros(200)
        self.y = np.zeros(200)

    def train(self, data_train, C=1, maxtime=1000, precision=1e-4):
        # Todo: train model
        X = data_train[:, :2]
        y = data_train[:, 2]
        self.X = X
        self.y = y
        self.alpha = np.zeros(len(X))
        self.E = np.zeros(len(X))
        training = True
        time = 0
        self.updateE(X, y)
    
        while training and time < maxtime:
            training = False
            time += 1
            i = self.find1()
            j = self.find2(i)
            print(time, i, j)
            if (y[i]*self.E[i] > precision and self.alpha[i] > precision) or (y[i]*self.E[i] < -precision and self.alpha[i] < C-precision) \
                or (abs(y[i]*self.E[i]-1) < precision and (self.alpha[i] < precision or self.alpha[i] > C-precision)):
                ita = kernel(X[i], X[i]) + kernel(X[j], X[j]) -  2*kernel(X[i], X[j])
                nalpha2 = self.alpha[j]+y[j]*(self.E[i]-self.E[j]) / ita
                if y[i] != y[j]:
                    L = max(0, self.alpha[j]-self.alpha[i])
                    H = min(C, C+self.alpha[j]-self.alpha[i])
                else:
                    L = max(0, self.alpha[j]-self.alpha[i]-C)
                    H = min(C, self.alpha[i]+self.alpha[j])
                # cut alpha2
                if nalpha2 > H:
                    nalpha2 = H
                elif nalpha2 < L:
                    nalpha2 = L
                nalpha1 = self.alpha[i] + y[i]*y[j]*(self.alpha[j] - nalpha2)
                if abs(self.alpha[j] - nalpha2) < 1e-4:
                    continue
                nb1 = -self.E[i] - y[i]*kernel(X[i], X[i])*(nalpha1 - self.alpha[i]) - y[j]*kernel(X[j], X[i])*(nalpha2-self.alpha[j])+self.b
                nb2 = -self.E[j] - y[i]*kernel(X[i], X[j])*(nalpha1 - self.alpha[i]) - y[j]*kernel(X[j], X[j])*(nalpha2-self.alpha[j])+self.b
                # choose new b
                if 0 < nalpha1 < C:
                    nb = nb1
                elif 0 < nalpha2 < C:
                    nb = nb2
                else:
                    nb = (nb1+nb2)/2
                # update params
                self.alpha[i] = nalpha1
                self.alpha[j] = nalpha2
                self.b = nb
                self.updateE(X, y)
                training = True

        return

    def predict(self, x):
        result = []
        for i in range(len(x)):
            t = 0
            for j in range(len(x)):
                t += self.alpha[j]*self.y[j]*kernel(x[i], self.X[j])
            t += self.b
            result.append(np.sign(t))
        return np.asarray(result).flatten()

    def cmptE(self, i, xtrain, y):
        e = 0
        for j in range(len(xtrain)):
            e += self.alpha[j] * y[j] * kernel(xtrain[i], xtrain[j])
        e = e + self.b - y[i]
        return e

    def updateE(self, xtrain, y):
        for i in range(len(xtrain)):
            self.E[i] = self.cmptE(i, xtrain, y)
    #randomly choose i
    def find1(self,):
        lst = []
        for i in range(len(self.E)):
            lst.append((self.E[i]+self.y[i])*self.y[i])
        return lst.index(min(lst))

    def find2(self, x):
        if self.E[x] > 0:
            return np.argmin(self.E)
        else:
            return np.argmax(self.E)


if __name__ == '__main__':
    # Load dataset
    train_file = 'data/train_kernel.txt'
    test_file = 'data/test_kernel.txt'
    # dataset format [x1, x2, t], shape (N * 3)
    data_train = load_data(train_file)
    data_test = load_data(test_file)

    # train SVM
    svm = SVM()
    svm.train(data_train,maxtime = 50, C=1)

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
