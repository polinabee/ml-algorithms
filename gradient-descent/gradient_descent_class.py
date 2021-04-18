import numpy as np
import matplotlib.pyplot as plt
import random


class GradientDescent:
    def __init__(self,
                 data_params=(0.4, 4, 1000),
                 step=0.1,
                 loss='hinge',
                 stochastic=True):
        self.mu, self.d, self.n = data_params
        self.data = self.generate_data()
        self.x_train, self.x_test, self.y_train, self.y_test = self.test_train_split()
        self.theta = np.random.uniform(0.0, 1.0, size=self.d)
        self.tuning_param = step
        self.phi = self.set_loss_function(loss)
        self.is_stochastic = stochastic

    def generate_data(self):
        mu, d, n = self.mu, self.d, self.n
        mean_a = [mu] + [0] * (d - 1)  # mean vector
        mean_b = [-mu] + [0] * (d - 1)
        cov = np.identity(d)  # identity covariance
        x_vals_a = np.random.multivariate_normal(mean_a, cov, int(n / 2))
        x_vals_b = np.random.multivariate_normal(mean_b, cov, int(n / 2))
        y_vals = np.random.choice([-1, 1], size=(n, 1))
        x_vals = np.concatenate((x_vals_a, x_vals_b), axis=0)
        return np.concatenate((x_vals, y_vals), axis=1)

    def test_train_split(self, split=0.7):
        data = self.data
        n = self.n
        x = np.asarray([a[:-1] for a in data])
        x_train, x_test = x[:int(n * split)], x[int(n * int(1 - split)):]

        y = np.asarray([a[-1] for a in data])
        y_train, y_test = y[:int(n * split)], y[int(n * int(1 - split)):]
        return x_train, x_test, y_train, y_test

    def set_loss_function(self, loss_name):
        loss_functions = {'hinge': lambda x: max(0, 1 + x),
                          'exp': lambda x: np.exp(x),
                          'logistic': lambda x: np.log2(1 + np.exp(x))}
        return loss_functions[loss_name] #, derivative of loss function

    def dot_product(self, x, w):
        return w.T @ x

    def RMSE(self, y):
        return np.sqrt(sum((y - self.y_test) ** 2) / len(y))

    def get_descent_rmse(self):
        return self.stochasticGradientDescent() if self.is_stochastic else self.gradientDescent()

    def stochasticGradientDescent(self, rmse_cutoff=0.7):
        x_train, x_text = self.x_train, self.x_test
        y_train, y_test = self.y_train, self.y_test
        phi = self.phi
        theta = self.theta

        RMSEs = []
        m = len(x_train)
        rmse = self.RMSE([self.dot_product(x, theta) for x in x_text])
        iterations = 0

        while rmse > rmse_cutoff and iterations < 15000:
            print(theta)
            iterations += 1
            i = random.randint(0, len(x_train) - 1)  # random point index
            for j in range(len(theta)):
                gradient = phi(self.dot_product(x_train[i], theta) - y_train[i]) * x_train[i][j]
                gradient *= 1 / m
                theta[j] = theta[j] - (self.tuning_param * gradient)
            y_pred = [self.dot_product(x, theta) for x in x_text]
            rmse = self.RMSE(y_pred)
            RMSEs.append(rmse)
        return RMSEs

    def gradientDescent(self, rmse_cutoff=0.7):
        x_train, x_text = self.x_train, self.x_test
        y_train, y_test = self.y_train, self.y_test
        phi = self.phi
        theta = self.theta

        RMSEs = []
        m = len(x_train)
        rmse = self.RMSE([self.dot_product(x, theta) for x in x_text])
        iterations = 0

        while rmse > rmse_cutoff and iterations < 3000:
            print(theta)
            iterations += 1
            for j in range(len(theta)):
                gradient = 0
                for i in range(m):
                    gradient += phi(self.dot_product(x_train[i], theta) - y_train[i]) * x_train[i][j]
                gradient *= 1 / m
                theta[j] = theta[j] - (self.tuning_param * gradient)
            y_pred = [self.dot_product(x, theta) for x in x_text]
            rmse = self.RMSE(y_pred, y_test)
            RMSEs.append(rmse)
        return RMSEs


def plot_descent(mu, d, n, step):
    for loss in ('hinge', 'exp', 'logistic'):
        gd = GradientDescent(data_params=(mu, d, n),
                             step=step,
                             loss=loss)
        error = gd.get_descent_rmse()
        plt.plot(error, label=loss)
    plt.legend()
    plt.title(f'RMSE convergence: {d}-dimensional data, {n} samples, tuning param: {step}')
    plt.savefig(f'sgd_{d}dim_size{n}_{step}step.png')


if __name__ == '__main__':
    mu = 0.4  # mean
    d = 4  # dimensions
    n = 1000  # number of points
    step = 0.1
    for dim in [3, 5, 10]:
        plot_descent(mu, dim, n, step)

    for sample_size in [200, 1000, 5000]:
        plot_descent(mu, d, sample_size, step)

    for step_size in [0.1, 0.01, 0.001]:
        plot_descent(mu, d, n, step_size)
