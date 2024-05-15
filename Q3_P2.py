import matplotlib.pyplot as plt
import numpy as np

# Set the actual dimensions of the figure
width = 6000
height = 3000

# Set the downscale factor for visualization
df = 10

# Calculate the downscaled dimensions
dscale_width = width // df
dscale_height = height // df

# Create a figure with the downscaled size
fig = plt.figure(figsize=(dscale_width/100, dscale_height/100), dpi=100)

# Add a subplot to the figure
ax = fig.add_subplot(111)

# Set the x-axis and y-axis limits based on the actual dimensions
ax.set_xlim(0, width)
ax.set_ylim(0, height)

# Set the x-axis and y-axis ticks based on the actual dimensions
ax.set_xticks(range(0, width+1, 1000))
ax.set_yticks(range(0, height+1, 500))

# Set the title of the figure
fig.suptitle(f'Figure ({width} x {height})')

# Define the coordinates and dimensions of the circles
circle1 = {'x': 1120, 'y': 2425, 'diameter': 800}
circle2 = {'x': 2630, 'y': 900, 'diameter': 1400}
circle3 = {'x': 4450, 'y': 2200, 'diameter': 750}

# Draw circle1
circle1_radius = circle1['diameter'] / 2
circle1_plot = plt.Circle((circle1['x'], circle1['y']), circle1_radius, color='red')
ax.add_patch(circle1_plot)

# Draw circle2
circle2_radius = circle2['diameter'] / 2
circle2_plot = plt.Circle((circle2['x'], circle2['y']), circle2_radius, color='green')
ax.add_patch(circle2_plot)

# Draw circle3
circle3_radius = circle3['diameter'] / 2
circle3_plot = plt.Circle((circle3['x'], circle3['y']), circle3_radius, color='blue')
ax.add_patch(circle3_plot)

# Define the coordinates of the start and goal points
start_point = {'x': 0, 'y': 1500}
goal_point = {'x': 6000, 'y': 1500}

# Plot the start point as a red diamond
start_point_plot = plt.plot(start_point['x'], start_point['y'], 'rD', markersize=5)

# Plot the goal point as a green star
goal_point_plot = plt.plot(goal_point['x'], goal_point['y'], 'rD', markersize=5)

# Tunable variables
robot_charge = 2.0
goal_charge = 5.0
obstacle_charge = 90.0
step_size = 50
goal_threshold = 50
obstacle_threshold = 100

# Function to calculate the attractive force from the goal
def attractive_force(robot_pos, goal_pos):
    direction = goal_pos - robot_pos
    distance = np.linalg.norm(direction)
    if distance <= goal_threshold:
        return np.zeros(2)
    force = goal_charge * robot_charge / (distance ** 2)
    return force * direction / distance

# Function to calculate the repulsive force from the obstacles
def repulsive_force(robot_pos, obstacle_pos, obstacle_radius):
    direction = robot_pos - obstacle_pos
    distance = np.linalg.norm(direction)
    if distance <= obstacle_radius + obstacle_threshold:
        if distance <= obstacle_radius:
            distance = obstacle_radius
        force = obstacle_charge * robot_charge / (distance ** 2)
        return force * direction / distance
    else:
        return np.zeros(2)

# Potential field planning algorithm
robot_pos = np.array([start_point['x'], start_point['y']], dtype=float)
path = [robot_pos]
exploration_nodes = []

max_iterations = 1000
iteration = 0

while np.linalg.norm(robot_pos - np.array([goal_point['x'], goal_point['y']])) > goal_threshold and iteration < max_iterations:
    # Calculate attractive force from the goal
    attractive_force_vec = attractive_force(robot_pos, np.array([goal_point['x'], goal_point['y']]))
    
    # Calculate repulsive forces from the obstacles
    repulsive_force_vec = np.zeros(2)
    for circle in [circle1, circle2, circle3]:
        repulsive_force_vec += repulsive_force(robot_pos, np.array([circle['x'], circle['y']]), circle['diameter'] / 2)
    
    # Calculate the resultant force
    resultant_force = attractive_force_vec + repulsive_force_vec
    
    # Normalize the resultant force
    if np.linalg.norm(resultant_force) > 0:
        resultant_force /= np.linalg.norm(resultant_force)
    
    # Move the robot in the direction of the resultant force with the specified step size
    new_robot_pos = robot_pos + step_size * resultant_force
    path.append(new_robot_pos)
    exploration_nodes.append(new_robot_pos)
    
    # Update the robot position
    robot_pos = new_robot_pos
    
    iteration += 1

# Plot the complete path
path_x, path_y = zip(*path)
plt.plot(path_x, path_y, 'b-', linewidth=2)


# Print the nodes in the final path
print("\nNodes in the Final Path:")
for i, node in enumerate(path):
    print(f"Node {i+1}: ({node[0]:.2f}, {node[1]:.2f})")

# Display the plot
plt.tight_layout()
plt.show()