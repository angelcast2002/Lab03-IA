import numpy as np
import matplotlib.pyplot as plt
import time

# Función polinómica
def polynomial(x):
    return 2 * x**3 - 3 * x**2 + 5 * x + 3

coeficientes_polinomio = [2, -3, 5, 3]
p = np.poly1d(coeficientes_polinomio)

# Función para obtener el costo. 
def cost_function(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def run_gradient_descent(X, y_true, coefficients, learning_rate, n_iterations):
    # Almacenar el tiempo de inicio
    start_time = time.time()

    # Almacenar los resultados para graficar
    cost_history = []

    # Gradiente estocástico
    for _ in range(n_iterations):
        for i in range(len(X)):
            # Calcula la predicción utilizando np.polyval
            y_pred = np.polyval(coefficients, X[i])

            # Calcula el gradiente de la función de costo con respecto a los coeficientes
            dp = np.polyder(p)
            gradient = dp(X[i]) * (y_pred - y_true[i])

            # Actualiza los coeficientes utilizando el gradiente y la tasa de aprendizaje
            coefficients = coefficients - learning_rate * gradient

        # Calcula el costo actual y almacénalo para graficar
        y_final = np.polyval(coefficients, X)
        current_cost = cost_function(y_true, y_final)
        cost_history.append(current_cost)

    # Almacenar el tiempo de finalización
    end_time = time.time()

    # Calcular el tiempo de ejecución
    execution_time = end_time - start_time

    # Muestra los coeficientes finales
    print("Coeficientes finales:", coefficients)

    # Imprimir el tiempo de ejecución
    print("Tiempo de ejecución:", execution_time, "segundos")

    # Grafica los datos reales y la predicción final
    plt.scatter(X, y_true, label='Datos reales')
    x_range = np.linspace(np.min(X), np.max(X), 100)
    y_pred = np.polyval(coefficients, x_range)
    plt.plot(x_range, y_pred, color='red', label='Polinomio ajustado')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Ajuste de polinomio con gradiente estocástico')

    # Ajustar los límites de la gráfica
    plt.xlim([np.min(X) - 1, np.max(X) + 1])  # Puedes ajustar estos valores según sea necesario
    plt.ylim([np.min(y_true) - 10, np.max(y_true) + 10])  # Ajusta el rango en y según tus datos

    plt.show()

# Datos de ejemplo
X = np.random.rand(100)
y_true = polynomial(X) + 0.1 * np.random.randn(100)

# Coeficientes iniciales
coefficients = np.random.rand(4)

# Tasa de aprendizaje y número de iteraciones
learning_rate = 0.01
n_iterations = 10000

# Llamar a la función principal
run_gradient_descent(X, y_true, coefficients, learning_rate, n_iterations)