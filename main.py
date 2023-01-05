"""
Definite integrals.
"""

import random

from sympy import integrate, init_printing, Symbol
import matplotlib.pyplot as plt

def riemann_method(func, a: float, b: float, n: int) -> float:
    """
    [a, b] - interval
    n - number of subintervals in the partition. points in them are chosen randomly
    func = cos(x), for example, where x = Symbol("x")
    Must return the approximate value of the integral of 'func' on [a, b]
    """
    result = 0
    delta_x = (b - a) / n

    for idx in range(0, n):
        ksi_i = a + idx * delta_x + random.uniform(0, delta_x)
        result += func.evalf(subs={x: ksi_i}) * delta_x

    return result

def trapezoidal_method(func, a: float, b: float, n: int) -> float:
    """
    [a, b] - interval
    n - number of subintervals in the partition (all should be the same size)
    func = 1 / x, for example, where x = Symbol("x")
    Must return the approximate value of the integral of 'func' on [a, b]
    """
    result = 0
    delta_x = (b - a) / n

    for idx in range(1, n + 1):
        x_i = a + delta_x * idx
        result += 0.5 * (func.evalf(subs={x: x_i}) + func.evalf(subs={x: x_i - delta_x})) * delta_x

    return result

def compare_accuracy(func, a: float, b: float, n: int):
    """
    Compares the real integral value and the one received from two upper functions.
    """
    real_value = integrate(func, (x, a, b)).evalf()
    riemann_method_value = riemann_method(func, a, b, n)
    trapezoidal_method_value = trapezoidal_method(func, a, b, n)

    return {
        "riemann_method_error": round(abs(real_value - riemann_method_value), 9),
        "trapezoidal_method_error": round(abs(real_value - trapezoidal_method_value), 9)
    }

def accuracy_table(func):
    """
    Builds table of with errors for every integration method.
    """
    _, ax = plt.subplots()

    n_list = [(10 ** n, f"10 ^ {n}") for n in range(0, 5)]
    col_labels = [n[1] for n in n_list]
    row_labels = ["Riemann method\nerror", "Trapezoidal method\nerror"]

    table_data = [[], []]
    for n in n_list:
        error = compare_accuracy(func, 0, 1, n[0])
        table_data[0].append(error["riemann_method_error"])
        table_data[1].append(error["trapezoidal_method_error"])

    table = ax.table(cellText=table_data, loc='center', colLabels=col_labels, rowLabels=row_labels)

    table.set_fontsize(17)
    table.scale(1, 3)
    ax.axis('off')

    plt.title(str(func))
    plt.show()

def get_plot_of_error(func, a: float, b: float):
    """
    Consider different n to see the error decrease, then plot the result
    """
    real_value = integrate(func, (x, a, b)).evalf()

    n_list = [ni for ni in range(1, 30)]

    delta_list = [abs(real_value - trapezoidal_method(func, a, b, ni)) for ni in n_list]
    ofunc_list = [ni ** (-2) for ni in n_list]

    constant = [df / of for of, df in zip(ofunc_list, delta_list)]
    constant = sum(constant) / len(constant)

    print(f"E(n) = {round(constant, 2)} * (1 / n ^ 2)")

    plt.title(str(func))
    plt.scatter(n_list, delta_list, color='red')
    plt.scatter(n_list, ofunc_list, color='green')

    plt.ylabel("Error function")
    plt.xlabel("green: 1 / x ^ 2\nred: error function")

    plt.show()

if __name__ == "__main__":
    x = Symbol("x")
    init_printing(unicode=True)
