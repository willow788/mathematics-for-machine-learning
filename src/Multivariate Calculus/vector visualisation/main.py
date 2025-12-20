import numpy as np
import matplotlib.pyplot as plt


#drawing the grid
x = np.linspace(-10, 10, 40)
y = np.linspace(-10, 10, 40)
X, Y = np.meshgrid(x,y)

#defining the vector field
U = -Y
V = X

#compute magnitude of vectors
magnitude = np.sqrt(U**2 + V**2)

#plot the vector field
plt.figure(figsize=(8,8))
plt.streamplot(X, Y, U, V, color=magnitude, cmap='viridis', linewidth=1, density=1.5)
plt.title('vector field visualization')


dt = 0.05
steps = 200
num_particles = 12

theta = np.linspace( 0, 2*np.pi, num_particles)
particles_x = 2*np.cos(theta)
particles_y = 2*np.sin(theta)

for _ in range(steps):
    px = particles_x
    py = particles_y

    vx = -py
    vy = px

    particles_x += vx * dt
    particles_y += vy * dt

    plt.plot(particles_x, particles_y, color='red', alpha=0.6)

    plt.title("vector field flow: F(x,y) = (-y, x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.show()


