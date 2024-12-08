import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#figure: first figure, a representation of particle movements in 2d rastrigin ----------------------------------------------

# Define the Rastrigin function
def rastrigin(x):
    return 10 * len(x) + sum(x_i**2 - 10 * np.cos(2 * np.pi * x_i) for x_i in x)

# Generate the Rastrigin function values
x_2d = np.linspace(-5.12, 5.12, 200)
y_2d = np.linspace(-5.12, 5.12, 200)
X, Y = np.meshgrid(x_2d, y_2d)
Z = np.array([rastrigin([x, y]) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

# Number of particles
num_particles = 30

# Random initial positions for particles
particles_x = np.random.uniform(-5, 5, num_particles)
particles_y = np.random.uniform(-5, 5, num_particles)

# Simulate random movements (as an example, not actual CBO algorithm)
movements_x = np.random.uniform(-1, 1, num_particles)
movements_y = np.random.uniform(-1, 1, num_particles)

# Final positions after movement
final_x = particles_x + movements_x
final_y = particles_y + movements_y

# Plot the Rastrigin contour
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.title('Rastrigin Function with Particle Movements')
plt.xlabel('x')
plt.ylabel('y')

# Plot the particles and their movements
for i in range(num_particles):
    plt.plot(particles_x[i], particles_y[i], 'ro')  # Initial positions
    plt.arrow(
        particles_x[i], particles_y[i],
        movements_x[i], movements_y[i],
        head_width=0.2, head_length=0.2, fc='black', ec='black'
    )  # Arrows indicating movement

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid(False)
plt.show()

#----------------------------------------------------------------------------------------------














#figure: rastrigin in 1d, 2d, 3d ----------------------------------------------------------------

# Define the Rastrigin function
def rastrigin(x):
    return 10 * len(x) + sum(x_i**2 - 10 * np.cos(2 * np.pi * x_i) for x_i in x)

# 1D Rastrigin
x_1d = np.linspace(-5.12, 5.12, 1000)
y_1d = [rastrigin([x]) for x in x_1d]

plt.figure(figsize=(10, 5))
plt.plot(x_1d, y_1d)
plt.title('Rastrigin Function in 1D')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()

# 2D Rastrigin
x_2d = np.linspace(-5.12, 5.12, 200)
y_2d = np.linspace(-5.12, 5.12, 200)
X, Y = np.meshgrid(x_2d, y_2d)
Z = np.array([rastrigin([x, y]) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.title('Rastrigin Function in 2D')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 3D Rastrigin
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_title('Rastrigin Function in 3D')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

plt.show()

#------------------------------------------------------------------------------------



