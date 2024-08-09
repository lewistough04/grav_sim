import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

G = 6.6743e-11
dt = 24 * 36000
steps = 100
colours = [
    'blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta',
    'yellow', 'pink', 'lime', 'teal', 'coral', 'navy'
]
# initial values
# could add UI for this in future
# Sun, mercury, venus, earth, mars, jupiter, saturn, uranus, pluto
# mass (kg)
# 1.989e30, 3.3e23, 4.87e24, 5.97e24, 6.42e23, 1.90e27, 5.68e26, 8.68e25, 1.02e26
# vel (m/s)
# 0, 4.787e4, 3.502e4, 2.978e4, 2.407e4, 1.307e4, 9.690e3, 6.810e3, 5.430e3
# distance from sun (m)
# 0, 5.79e10, 1.08e11, 1.50e11, 2.28e11, 7.78e11, 1.43e12, 2.87e12, 4.50e12
masses = np.array([1.989e30,3.3e23, 4.87e24, 5.97e24, 6.42e23, 1.90e27, 5.68e26, 8.68e25, 1.02e26], dtype=np.float64)
velocities = np.array([[0,0,0], [0, 4.787e4,0], [0,3.502e4,0], [0,2.978e4,0], [0,2.407e4,0], [0,1.307e4,0], [0,9.690e3,0], [0,6.810e3,0], [0,5.430e3,0]], dtype=np.float64)
positions = np.array([[0,0,0], [5.79e10, 0,0], [1.08e11, 0,0], [1.50e11, 0,0], [2.28e11, 0,0], [7.78e11, 0,0], [1.43e12, 0,0], [2.87e12, 0,0],[4.50e12, 0,0]], dtype=np.float64)


num_masses = len(masses)
trail_length = 100
trails = [np.zeros((trail_length, 3), dtype=np.float64) for _ in range(num_masses)]
# Euler method
def forces_calc(positions, masses):
    forces = np.zeros_like(positions, dtype=np.float64)
    for i in range(num_masses):
        for j in range(num_masses):
            if i != j:
                r_ij = positions[j] - positions[i]  # vector between i and j
                dist = np.linalg.norm(r_ij)         # distance of vector
                if dist > 0:
                    forces[i] += G * masses[i] * masses[j] / dist**3 * r_ij
    return forces

def update_pos_vel(positions, velocities, masses):
    forces = forces_calc(positions, masses)
    velocities += (forces / masses[:,None]) * dt
    positions += velocities * dt
    return positions, velocities


# plot in matplotlib

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

scat = ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=colours[:len(masses)], s=10)
lines = [ax.plot([], [], [], color=colours[i])[0] for i in range(num_masses)]

ax.set_xlim(-1e12, 1e12)
ax.set_ylim(-1e12, 1e12)
ax.set_zlim(-1e12, 1e12)
ax.set_xlabel('X Pos (m)')
ax.set_ylabel('Y Pos (m)')
ax.set_zlabel('Z Pos (m)')
ax.set_title('N-Body Sim')


def update(frame):
    global positions, velocities, trails
    positions, velocities = update_pos_vel(positions, velocities, masses)
    scat._offsets3d = (positions[:,0],positions[:,1],positions[:,2])
    for i in range(num_masses):
        trails[i] = np.vstack((trails[i][1:], positions[i]))
        lines[i].set_data(trails[i][:, 0], trails[i][:, 1])
        lines[i].set_3d_properties(trails[i][:, 2])
    return scat, *lines

ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=False)
plt.show()