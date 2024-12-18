import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define A(x), its derivative, and parameters
def A(x):
    """Constraint matrix A(x)."""
    # Replace this with your actual function
    theta = x[2].item()
    return torch.tensor([[-np.sin(theta), np.cos(theta), 0]], dtype=torch.float32)

def dAdx(x):
    """Derivative of A(x) with respect to x."""
    # Example: Replace this with your actual derivative , return c*n*n tensor (last dimension -> derivative through x)
    theta = x[2].item()
    c = np.cos(theta)
    s = np.sin(theta)
    return torch.tensor([[[0,0,-c],[0,0,-s],[0,0,0]]], dtype=torch.float32)


# Define the ODE system
def ode_system(t, y, Q, x_dim):
    """Compute the derivatives for x and lambda."""
    x = y[:x_dim]
    lam = y[x_dim:]
    
    # Constraint matrix and its derivative
    A_x = A(x)
    dA_x = dAdx(x)
    AQ_inv_AT = A_x @ torch.linalg.inv(Q) @ A_x.T
    P = -torch.linalg.inv(Q) + torch.linalg.inv(Q)@A_x.T @ torch.linalg.inv(AQ_inv_AT) @ A_x @ torch.linalg.inv(Q)
    # Compute derivatives
    u = P @ lam
    mu = -torch.linalg.inv(AQ_inv_AT) @ A_x @ torch.linalg.inv(Q) @ lam
    u = u.squeeze()
    if mu.shape == (1,1):
        mu = mu.reshape(1)
    elif mu.shape == (1,):
        mu = mu
    else:
        mu = mu.squeeze()
    dx_dt = u
    dlambda_dt = -torch.einsum('j,jki,k->i',mu,dA_x,u)
    return torch.cat((dx_dt, dlambda_dt),dim=0)

def wrap_angle(theta):
    """Wrap an angle to [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi

def distance(q1, q2, Q):
    """Compute the weighted distance between q1 and q2."""
    dx = q1[0] - q2[0]
    dy = q1[1] - q2[1]
    dtheta = wrap_angle(q1[2] - q2[2])
    dq = torch.tensor([dx,dy,dtheta])
    return dq.T @ Q @ dq

# Boundary value shooting function
def shooting_error(lam0, x0, xT, T, Q):
    """Error function for the shooting method."""
    lam0 = torch.tensor(lam0, dtype=torch.float32) if isinstance(lam0, np.ndarray) else lam0

    x_dim = len(x0)
    y0 = torch.cat((x0, lam0),dim=0)

    # Solve the ODE
    sol = solve_ivp(
        ode_system, [0, T], y0, args=(Q, x_dim),
        method='RK45', t_eval = [T], vectorized = True
    )
    
    # Extract x(T)
    xT_actual = sol.y[:x_dim, -1]
    # print(xT_actual)
    # print(xT)

    #Change distance metric as your application
    dist = distance(xT, xT_actual,Q)
    # Compute the error
    print(dist.item())
    return dist

# Main function to solve the problem
def solve_optimal_control(x0, xT, T, Q):
    """Solve the ODE using the shooting method."""
    x_dim = len(x0)
    
    # Initial guess for lambda(0)
    lam0_guess = torch.randn(x_dim) * 0.1
    
    # Optimize lambda(0) to minimize the shooting error
    # result = minimize(
    #     lambda lam0: shooting_error(lam0, x0, xT, T, Q),
    #     lam0_guess,
    #     method='Powell'
    #     # method ='Nelder-Med'
    # )

    bounds = [(-10, 10)] * x_dim
    result = differential_evolution(
        lambda lam0: shooting_error(lam0, x0, xT, T, Q),
        bounds,
        strategy="best1bin",
        atol=1e-6
    )
    
    if result.success:
        lam0_opt = result.x
        print(f"Optimal lambda(0) found: {lam0_opt}")
        lam0_opt = torch.tensor(lam0_opt, dtype=torch.float32) if isinstance(lam0_opt, np.ndarray) else lam0_opt
        # Solve the ODE with the optimal lambda(0)
        y0 = torch.cat((x0, lam0_opt),dim=0)
        sol = solve_ivp(
            ode_system, [0, T], y0, args=(Q, x_dim),
            method='RK45', t_eval=np.linspace(0, T, 100)
        )
        return sol
    else:
        raise RuntimeError("Shooting method failed to converge.")

def visualize_trajectory(t_vals, x_vals):
    """Visualize the trajectory in 2D space."""
    x_positions = x_vals[:, 0]  # x(0): x-coordinate
    y_positions = x_vals[:, 1]  # x(1): y-coordinate
    orientations = x_vals[:, 2]  # x(2): orientation angles
    
    # Plot trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(x_positions, y_positions, label="Trajectory", color="blue", lw=2)
    
    # Add quiver arrows for orientation
    skip = max(1, len(t_vals) // 24)  # Skip some points for clarity
    u = 0.005*np.cos(orientations)  # x-component of orientation vector
    v = 0.005*np.sin(orientations)  # y-component of orientation vector

    plt.quiver(
        x_positions[::skip], y_positions[::skip],
        u[::skip], v[::skip],
        angles="xy", scale_units="xy", scale=0.1, color="red", alpha=0.8,
        label="Orientation"
    )
    
    # Highlight start and end points
    plt.scatter(x_positions[0], y_positions[0], color="green", label="Start", zorder=5)
    plt.scatter(x_positions[-1], y_positions[-1], color="orange", label="End", zorder=5)
    
    # Add labels and legend
    plt.title("2D Trajectory and Orientation")
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plot_path = 'plots_opt/trajectory2D.png'
    plt.savefig(plot_path)
    plt.close()

def implement_optimal_trajectory(x0, T, Q):
    x_dim = len(x0)
    
    nu = Q[2,2]

    if nu == 0.01:
        lam0_opt = torch.tensor([1.02550599e-04, -1.99922897e+00,  3.77254239e+00]) #used Q_33 = 0.01
    elif nu == 1 :
        lam0_opt = torch.tensor([0.00874898, -6.80807242, -3.62507805]) #used Q_33 = 1
    elif nu == 0.1 :
        lam0_opt = torch.tensor([-5.99159136e-03, -8.46586907e+00,  1.28506257e+00]) #used Q_33 = 0.1
    else: 
        lam0_opt = torch.tensor([-2.1403, -8.2920,  0.2049])
    
    print(f"Optimal lambda(0) used: {lam0_opt}")
    # Solve the ODE with the optimal lambda(0)
    y0 = torch.cat((x0, lam0_opt),dim=0)
    sol = solve_ivp(
        ode_system, [0, T], y0, args=(Q, x_dim),
        method='RK45', t_eval=np.linspace(0, T, 100)
    )
    return sol

def vis_3d(t_vals, x_vals):
    """
    Visualize the trajectory in R^2 x S^1 manifold.
    
    Parameters:
        t_vals torch.tensor
        x_vals tenosr of trajectory points, each containing [x, y, theta].
    """
    if x_vals.shape[1] != 3:
        raise ValueError("x_vals must have shape (N, 3), where N is the number of points.")
    
    # Extract x, y, and theta from x_vals
    x = x_vals[:, 0]
    y = x_vals[:, 1]
    theta = x_vals[:, 2]

    # Prepare the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory in R^2 x S^1
    ax.plot(x_vals[:, 0], x_vals[:, 1], x_vals[:, 2], label="Trajectory", color="b")

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Theta')

    # Set title
    ax.set_title('Trajectory on R^2 x S^1 Manifold')

    # Show plot
    plot_path = 'plots_opt/trajectory3D.png'
    plt.savefig(plot_path)
    plt.close()


# Example usage
if __name__ == "__main__":
    # Define parameters
    x0 = torch.tensor([0.0, 0.0, 0.0])  # Initial state
    xT = torch.tensor([0.0, 1.0, 0.0])  # Final state
    T = 1.0  # Final time
    # Q = torch.tensor([[1,0,0],[0,1,0],[0,0,0.01]])
    nu = 1.0
    Q = torch.tensor([[1,0,0],[0,1,0],[0,0,nu]])

    '''
    for optimization...
    '''

    # Solve the system
    # solution = solve_optimal_control(x0, xT, T, Q)

    '''
    for deployment...
    '''

    # Use this to implement trajectory w/o optimization
    solution = implement_optimal_trajectory(x0, T, Q)
    

    # Extract solution
    t_vals = solution.t
    x_vals = solution.y[:len(x0), :].T
    lam_vals = solution.y[len(x0):, :].T

    # Print and plot results
    print("Solution:")
    print("Time:", t_vals)
    print("States:", x_vals)
    print("Costates:", lam_vals)
    
    visualize_trajectory(t_vals, x_vals)
    vis_3d(t_vals, x_vals)

