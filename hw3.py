import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as plt
import time


def plot_graph(figure_title, iter, convergence_curve, save_name):
    plt.plot(range(iter), convergence_curve)
    plt.title(figure_title)
    plt.ylabel(r"$f(x_{k})-f^*$")
    plt.xlabel("Iteration Number")
    plt.yscale('log')
    plt.savefig('graphs/' + save_name + '.svg', format='svg')
    plt.show()


def rosen(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der


def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
    H = H + np.diag(diagonal)
    return H


def gradient_decent(f, der_f, x0, alpha0, sigma, beta, epsilon, figure_title, save_name):
    start = time.time()
    d = -der_f(x0)
    x = x0
    iter = 0
    convergence_curve = []
    while np.linalg.norm(der_f(x)) >= epsilon:
        convergence_curve.append(f(x))
        alpha = alpha0
        F_armijo = (f(x + alpha * d) - f(x))
        F_armijo_sigma = (sigma * alpha * np.matmul(der_f(x).T, d))
        while F_armijo > F_armijo_sigma:
            alpha = beta * alpha
            F_armijo = (f(x + alpha * d) - f(x))
            F_armijo_sigma = (sigma * alpha * np.matmul(der_f(x).T, d))
        x = x + alpha * d
        d = -der_f(x)
        iter += 1
    end = time.time()
    plot_graph(figure_title + "\nTotal running time: " + str(("{:.5f}".format(end - start))) + " sec", iter,
               convergence_curve, save_name)
    return x


def quasi_newton(f, der_f, x0, alpha0, sigma, beta, epsilon, figure_title, save_name):
    start = time.time()
    grad = der_f(x0)
    x = x0
    iter = 0
    B = np.identity(len(x0))
    convergence_curve = []
    while np.linalg.norm(der_f(x)) >= epsilon:
        d = np.matmul(-B, grad)
        # print(np.linalg.norm(der_f(x)) - epsilon)
        convergence_curve.append(f(x))
        alpha = alpha0
        F_armijo = (f(x + alpha * d) - f(x))
        F_armijo_sigma = (sigma * alpha * np.matmul(der_f(x).T, d))
        while F_armijo > F_armijo_sigma:
            alpha = beta * alpha
            F_armijo = (f(x + alpha * d) - f(x))
            F_armijo_sigma = (sigma * alpha * np.matmul(der_f(x).T, d))
        prev_x = x
        prev_grad = grad
        x = x + alpha * d
        grad = der_f(x)
        p = x - prev_x
        q = grad - prev_grad
        v = p - np.matmul(B, q)
        v = np.reshape(v, (10, 1))
        # prevent vanishing denominator
        if np.matmul(v.T, q) != 0:
            B = B + np.matmul(v, v.T) / np.matmul(v.T, q)
            """H = np.linalg.inv(rosen_hess(x))
            print(B - H)"""
        else:
            B = np.identity(len(x0))
        iter += 1
    end = time.time()
    plot_graph(figure_title + "\nTotal running time: " + str(("{:.5f}".format(end - start))) + " sec", iter,
               convergence_curve, save_name)
    return x


def BFGS(f, der_f, x0, alpha0, sigma, beta, epsilon, figure_title, save_name):
    start = time.time()
    c2 = 0.9
    grad = der_f(x0)
    x = x0
    iter = 0
    B = np.identity(len(x0))
    convergence_curve = []
    while np.linalg.norm(der_f(x)) >= epsilon:
        d = np.matmul(-B, grad)
        # print(np.linalg.norm(der_f(x)) - epsilon)
        convergence_curve.append(f(x))
        alpha = alpha0
        F_armijo = (f(x + alpha * d) - f(x))
        F_armijo_sigma = (sigma * alpha * np.matmul(der_f(x).T, d))
        while F_armijo > F_armijo_sigma:
            alpha = beta * alpha
            F_armijo = (f(x + alpha * d) - f(x))
            F_armijo_sigma = (sigma * alpha * np.matmul(der_f(x).T, d))
        prev_x = x
        prev_grad = grad
        x = x + alpha * d
        grad = der_f(x)
        p = x - prev_x
        q = grad - prev_grad
        s = np.matmul(B, q)
        tau = np.matmul(s.T, q)
        mu = np.matmul(p.T, q)
        v = 1 / mu * p - 1 / tau * s
        v = np.reshape(v, (10, 1))
        s = np.reshape(s, (10, 1))
        p = np.reshape(p, (10, 1))
        if np.matmul(grad.T, d) > c2 * np.matmul(prev_grad.T, d):
            B = B + 1 / mu * np.matmul(p, p.T) - 1 / tau * np.matmul(s, s.T) + tau * np.matmul(v, v.T)
            """H = np.linalg.inv(rosen_hess(x))
            print(B - H)"""
        iter += 1
    end = time.time()
    plot_graph(figure_title + "\nTotal running time: " + str(("{:.5f}".format(end - start))) + " sec", iter,
               convergence_curve, save_name)
    return x


if __name__ == '__main__':
    x0 = np.zeros(10)
    alpha0 = 1
    sigma = 0.25
    beta = 0.5
    epsilon = 1e-5
    val = gradient_decent(rosen, rosen_der, x0, alpha0, sigma, beta, epsilon,
                          figure_title="Gradient Descent", save_name="gd")
    print(rosen(val))
    val = quasi_newton(rosen, rosen_der, x0, alpha0, sigma, beta, epsilon,
                       figure_title="Quasi Newton", save_name="quasi")
    print(rosen(val))
    val = BFGS(rosen, rosen_der, x0, alpha0, sigma, beta, epsilon,
               figure_title="BFGS", save_name="BFGS")
    print(rosen(val))
