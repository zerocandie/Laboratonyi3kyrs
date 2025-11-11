import numpy
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def f(x):
    A1, A2 = 2,1
    a1, a2, b1, b2, c1, c2, d1, d2 = 3,3,2,1,2,1,1,3
    return A1 * numpy.exp(-(((x[0]-a1)/b1)**2)-(((x[1]-c1)/d1))**2) + A2 * numpy.exp(-(((x[0]-a2)/b2)**2) - (((x[1]-c2)/d2)**2))
def grad_f(x):
        h = 1e-8
        df_dx = (f(x+numpy.array([h,0]))- f(x)) / h
        df_dy = (f(x+numpy.array([0,h]))-f(x)) / h
        return numpy.array([df_dx,df_dy])

def steepest_descent(x0, epsilon=1e-6):
    x = numpy.array(x0)
    trajectory = [x.tolist()]
    while True:
        gradient = grad_f(x)
        direction = gradient  # для подъёма
        line_search_result = minimize(lambda alpha: -f(x + alpha * direction), x0=0, bounds=[(0, None)])
        alpha = line_search_result.x[0]
        x_new = x + alpha * direction
        if numpy.linalg.norm(x_new - x) < epsilon:
            break
        x = x_new
        trajectory.append(x.tolist())
    return x, trajectory  # ← вне цикла!
      
def huka(x0, learning_rate=0.1, epsilon=1e-6, max_iterations=100000):
      x = numpy.array(x0)
      trajectory = [x.tolist()]

      for _ in range (max_iterations):
            gradient = grad_f(x)
            x_new = x + learning_rate * gradient
            if numpy.linalg.norm(x_new - x) < epsilon:
                  break
            x = x_new
            trajectory.append(x.tolist())
      return x,trajectory
x0_1 = [0 , 0]
x0_2 = [2, 2]
epsilon_steepest_descent = 0.01

# Вызов метода наискорейшего спуска для первой точки
result_x_sd_1, trajectory_sd_1 = steepest_descent(x0_1, epsilon_steepest_descent)
# Вызов метода наискорейшего спуска для второй точки
result_x_sd_2, trajectory_sd_2 = steepest_descent(x0_2, epsilon_steepest_descent)
# Вызов метода Хука и Дживса для первой точки
learning_rate = 0.1
result_x_gd_1, trajectory_gd_1 = huka(x0_1, learning_rate)
# Вызов метода Хука и Дживса для второй точки
result_x_gd_2, trajectory_gd_2 = huka(x0_2, learning_rate)
# Визуализация


x_values, y_values = numpy.meshgrid(numpy.linspace(-5, 5, 100), numpy.linspace(-5, 5, 100))
z_values = numpy.zeros_like(x_values)
for i in range(x_values.shape[0]):
    for j in range(x_values.shape[1]):
        z_values[i, j] = f([x_values[i, j], y_values[i, j]])
plt.contour(x_values, y_values, z_values, levels=20, cmap='viridis')
trajectory_sd_1 = numpy.array(trajectory_sd_1).T
trajectory_gd_1 = numpy.array(trajectory_gd_1).T
plt.plot(trajectory_sd_1[0], trajectory_sd_1[1], marker='o', color='blue', label='Метод наискорейшего спуска (1)')
plt.plot(trajectory_gd_1[0], trajectory_gd_1[1], marker='x', color='red', label='метод Хука и Дживса (1)')
trajectory_sd_2 = numpy.array(trajectory_sd_2).T
trajectory_gd_2 = numpy.array(trajectory_gd_2).T
plt.plot(trajectory_sd_2[0], trajectory_sd_2[1], marker='s', color='green', label='Метод наискорейшего спуска (2)')
plt.plot(trajectory_gd_2[0], trajectory_gd_2[1], marker='^', color='purple', label='метод Хука и Дживса (2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
