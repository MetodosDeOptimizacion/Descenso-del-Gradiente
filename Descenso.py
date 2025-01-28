import numpy as np
import matplotlib.pyplot as plt

# Ejercicio 1: Mínimo de una función cuadrática en 1D
def ejercicio_1():
    g = lambda x: (x - 5)**2
    dg_dx = lambda x: 2 * (x - 5)
    
    x_values = [10]  # Punto inicial
    eta = 0.2
    iterations = 5
    
    for _ in range(iterations):
        x_next = x_values[-1] - eta * dg_dx(x_values[-1])
        x_values.append(x_next)
    
    x_values = np.array(x_values)
    g_values = g(x_values)
    
    plt.figure()
    plt.plot(range(iterations + 1), g_values, marker='o')
    plt.title('Ejercicio 1: Minimización de g(x)')
    plt.xlabel('Iteración')
    plt.ylabel('g(x)')
    plt.grid()
    plt.show()

# Ejercicio 2: Ajuste de una recta
def ejercicio_2():
    x_i = np.array([1, 2, 3, 4, 5])
    y_i = np.array([2, 2.8, 3.6, 4.5, 5.1])
    eta = 0.01
    iterations = 3

    beta_0, beta_1 = 0, 0
    costs = []
    
    for _ in range(iterations):
        errors = y_i - (beta_0 + beta_1 * x_i)
        grad_beta_0 = -2 * np.sum(errors) / len(x_i)
        grad_beta_1 = -2 * np.sum(errors * x_i) / len(x_i)
        
        beta_0 -= eta * grad_beta_0
        beta_1 -= eta * grad_beta_1
        
        cost = np.mean(errors**2)
        costs.append(cost)
    
    plt.figure()
    plt.plot(range(1, iterations + 1), costs, marker='o')
    plt.title('Ejercicio 2: Reducción de Costo')
    plt.xlabel('Iteración')
    plt.ylabel('Costo J(β0, β1)')
    plt.grid()
    plt.show()

# Ejercicio 3: Clasificación logística
def ejercicio_3():
    data = np.array([[0.5, 1.0, 0], [1.5, 2.0, 0], [2.0, 2.5, 1], [3.0, 3.5, 1]])
    x = data[:, :2]
    y = data[:, 2]
    eta = 0.1
    iterations = 3

    w = np.zeros(3)
    costs = []
    
    for _ in range(iterations):
        z = w[0] + np.dot(x, w[1:])
        predictions = 1 / (1 + np.exp(-z))
        errors = predictions - y
        
        grad_w0 = np.mean(errors)
        grad_w1 = np.mean(errors * x[:, 0])
        grad_w2 = np.mean(errors * x[:, 1])
        
        w[0] -= eta * grad_w0
        w[1] -= eta * grad_w1
        w[2] -= eta * grad_w2
        
        cost = -np.mean(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8))
        costs.append(cost)
    
    plt.figure()
    plt.plot(range(1, iterations + 1), costs, marker='o')
    plt.title('Ejercicio 3: Reducción de Costo Logístico')
    plt.xlabel('Iteración')
    plt.ylabel('Costo')
    plt.grid()
    plt.show()

# Ejercicio 4: Descenso estocástico en minibatches
def ejercicio_4():
    np.random.seed(42)
    x_data = np.random.rand(1000, 3)
    y_data = np.dot(x_data, [3, -2, 1]) + np.random.normal(0, 0.1, 1000)
    
    batch_size = 50
    eta = 0.01
    w = np.zeros(3)
    iterations = 3
    costs = []
    
    for _ in range(iterations):
        for i in range(0, len(x_data), batch_size):
            batch_x = x_data[i:i + batch_size]
            batch_y = y_data[i:i + batch_size]
            
            predictions = np.dot(batch_x, w)
            errors = predictions - batch_y
            
            grad_w = (2 / batch_size) * np.dot(batch_x.T, errors)
            w -= eta * grad_w
        
        cost = np.mean((np.dot(x_data, w) - y_data)**2)
        costs.append(cost)
    
    plt.figure()
    plt.plot(range(1, iterations + 1), costs, marker='o')
    plt.title('Ejercicio 4: Reducción de Costo en Minibatches')
    plt.xlabel('Iteración')
    plt.ylabel('Costo')
    plt.grid()
    plt.show()

# Llamar a las funciones para resolver los ejercicios y graficar
ejercicio_1()
ejercicio_2()
ejercicio_3()
ejercicio_4()
