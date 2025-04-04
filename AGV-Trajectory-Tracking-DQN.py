import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
import random

# AGV Dynamics Model (Unicycle Model)
def agv_dynamics(state, t, left_wheel_speed, right_wheel_speed):
    x, y, theta = state
    L = 0.5  # Wheelbase length
    dxdt = (left_wheel_speed + right_wheel_speed) * np.cos(theta) / 2
    dydt = (left_wheel_speed + right_wheel_speed) * np.sin(theta) / 2
    dthetadt = (right_wheel_speed - left_wheel_speed) / L
    return [dxdt, dydt, dthetadt]

# Trajectory Planning 
def reference_trajectory(t, radius=5.0):
    x_ref = radius * np.cos(t)
    y_ref = radius * np.sin(t)
    return x_ref, y_ref

# DQN Agent
class DQNAgent(nn.Module):
    def __init__(self, state_space_size, action_space_size, learning_rate=0.001, discount_factor=0.99):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_space_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space_size)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 64

        # Target model with identical architecture
        self.target_model = nn.Sequential(
            nn.Linear(state_space_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size)
        )
        self.update_target_model()  # Initialize target model weights

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        mini_batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        Q_values = self(states).gather(1, actions)
        next_Q_values = self.target_model(next_states).max(1, keepdim=True)[0]
        target_Q_values = rewards + self.discount_factor * next_Q_values * (1 - dones)

        loss = nn.MSELoss()(Q_values, target_Q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        """Synchronize the target model weights with the main model."""
        state_dict_main = {k: v for k, v in self.state_dict().items() if not k.startswith("target_model")}
        self.target_model.load_state_dict(state_dict_main)

    def get_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.fc3.out_features - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            Q_values = self(state)
        return Q_values.argmax().item()

# PI Controller for Speed Control
class PIController:
    def __init__(self, kp, ki, saturation=None):
        self.kp = kp
        self.ki = ki
        self.integral = 0
        self.saturation = saturation

    def control(self, error):
        self.integral += error
        control_signal = self.kp * error + self.ki * self.integral

        if self.saturation is not None:
            control_signal = max(min(control_signal, self.saturation[1]), self.saturation[0])

        return control_signal

# Simulation parameters
total_time = 20
dt = 0.1
num_steps = int(total_time / dt)

# Initialize AGV state
agv_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]

# Initialize controllers
state_space_size = 3
action_space_size = 5
dqn_agent = DQNAgent(state_space_size=state_space_size, action_space_size=action_space_size)
pi_controller_left = PIController(kp=1.0, ki=0.1, saturation=(-1.0, 1.0))
pi_controller_right = PIController(kp=1.0, ki=0.1, saturation=(-1.0, 1.0))

# Precompute reference trajectory
time_points = np.linspace(0, total_time, num_steps)
x_ref_trajectory, y_ref_trajectory = reference_trajectory(time_points)

# Lists to store simulation data
trajectory_data = []

# Simulation loop
for step in range(num_steps):
    t = step * dt

    # Get reference trajectory at this time step
    x_ref, y_ref = x_ref_trajectory[step], y_ref_trajectory[step]

    # Calculate errors
    error_x = x_ref - agv_state[0]
    error_y = y_ref - agv_state[1]

    # Calculate DQN control action
    rl_action = dqn_agent.get_action(agv_state)

    # Calculate PI control for left and right wheel speeds
    left_wheel_speed = pi_controller_left.control(error_x)
    right_wheel_speed = pi_controller_right.control(error_y)

    # Apply control signals to AGV dynamics model
    ode_args = (left_wheel_speed, right_wheel_speed)
    agv_state = odeint(agv_dynamics, agv_state, [t, t + dt], args=ode_args)[-1]

    # Update DQN memory and train the agent
    reward = -np.sqrt(error_x**2 + error_y**2)  # Negative reward for deviation
    dqn_agent.remember(agv_state.tolist(), rl_action, reward, agv_state.tolist(), False)
    dqn_agent.train()

    # Store simulation data
    trajectory_data.append(agv_state[:2])

# Convert simulation data to numpy array
trajectory_data = np.array(trajectory_data)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(trajectory_data[:, 0], trajectory_data[:, 1], label='AGV Trajectory')
plt.plot(x_ref_trajectory, y_ref_trajectory, label='Reference Trajectory', linestyle='--')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Improved AGV Trajectory Tracking with DQN')
plt.grid()
plt.show()
