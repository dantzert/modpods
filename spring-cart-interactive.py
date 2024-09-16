# not at all working

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp
import ipywidgets as widgets
from ipywidgets import interact

# Parameters
m_cart = 1.0   # Mass of the cart (kg)
m_pend = 0.1   # Mass of the pendulum (kg)
l_pend = 1.0   # Length of the pendulum (m)
g = 9.81       # Gravitational acceleration (m/s^2)
damping = 0.1  # Damping factor

# Control Parameters (PD Controller)
Kp = 100.0  # Proportional gain
Kd = 20.0   # Derivative gain

# Linearized system dynamics
def linear_system(t, state, stiffness, control_input, disturbance):
    x, x_dot, theta, theta_dot = state
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Control input through spring
    u = -Kp * theta - Kd * theta_dot + control_input
    u_spring = stiffness * u
    
    # Equations of motion (linearized)
    x_ddot = (u_spring + disturbance - m_pend * l_pend * theta_dot**2 * sin_theta + m_pend * g * sin_theta * cos_theta) / \
             (m_cart + m_pend * sin_theta**2)
    theta_ddot = (g * sin_theta - cos_theta * x_ddot) / l_pend
    
    return [x_dot, x_ddot, theta_dot, theta_ddot]

# Solve the system with initial conditions
def solve_pendulum(stiffness, control_input, disturbance, t_max=10):
    t_span = (0, t_max)
    y0 = [0, 0, np.pi / 12, 0]  # Initial conditions: [x, x_dot, theta, theta_dot]
    t_eval = np.linspace(0, t_max, 300)
    
    # Integrate the system
    sol = solve_ivp(linear_system, t_span, y0, args=(stiffness, control_input, disturbance), t_eval=t_eval, method='RK45')
    return sol.t, sol.y

# Animate the system
def animate_pendulum(stiffness, control_input, disturbance):
    t, y = solve_pendulum(stiffness, control_input, disturbance)
    x = y[0, :]   # Cart position
    theta = y[2, :]  # Pendulum angle
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 2)
    
    # Elements of the system: cart and pendulum
    cart, = ax.plot([], [], 'o-', lw=2)
    pendulum, = ax.plot([], [], 'o-', lw=2)
    
    def init():
        cart.set_data([], [])
        pendulum.set_data([], [])
        return cart, pendulum
    
    def update(frame):
        cart.set_data([x[frame] - 0.5, x[frame] + 0.5], [0, 0])  # Draw cart
        pendulum.set_data([x[frame], x[frame] + l_pend * np.sin(theta[frame])],
                          [0, -l_pend * np.cos(theta[frame])])  # Draw pendulum
        return cart, pendulum
    
    # Slow down animation
    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=100)  # Interval adjusted to 100ms
    
    plt.title('Inverted Pendulum on Cart')
    plt.grid(True)
    plt.show()

# Interactive widgets for spring stiffness, control input, and disturbance
stiffness_slider = widgets.FloatSlider(min=0.1, max=10.0, step=0.1, value=1.0, description="Spring Stiffness")
control_input_slider = widgets.FloatSlider(min=-10.0, max=10.0, step=0.1, value=0.0, description="Control Input")
disturbance_slider = widgets.FloatSlider(min=-5.0, max=5.0, step=0.1, value=0.0, description="Disturbance")

# Ensure ipywidgets work in Jupyter by using `interact`
if 'get_ipython' in globals():  # Checks if running in a Jupyter environment
    interact(animate_pendulum,
             stiffness=stiffness_slider,
             control_input=control_input_slider,
             disturbance=disturbance_slider)
else:
    # If not running in Jupyter, manually call animation for default parameters
    animate_pendulum(1.0, 0.0, 0.0)
