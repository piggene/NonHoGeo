import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import argparse
from scipy.integrate import solve_ivp
from model import XtolamNet
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchdiffeq import odeint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ODESystem:
    def __init__(self):
        self.x0 = torch.tensor([0.0, 0.0, 0.0], device = device)  # Initial state
        self.T = 1.0  # Final time
        self.Q = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0.1]], dtype=torch.float32, device = device)

    def A_cpu(self, x):
        """Constraint matrix A(x)."""
        theta = x[2].item()
        return torch.tensor([[-np.sin(theta), np.cos(theta), 0]], dtype=torch.float32, device = device)

    def dAdx_cpu(self, x):
        """Derivative of A(x) with respect to x."""
        theta = x[2].item()
        c = np.cos(theta)
        s = np.sin(theta)
        return torch.tensor([[[0, 0, -c], [0, 0, -s], [0, 0, 0]]], dtype=torch.float32, device = device)

    def ode_system_cpu(self, t, y, Q, x_dim):
        """Compute the derivatives for x and lambda."""
        x = torch.tensor(y[:x_dim], dtype=torch.float32, device = device)
        lam = torch.tensor(y[x_dim:], dtype=torch.float32, device = device)

        # Constraint matrix and its derivative
        A_x = self.A_cpu(x)
        dA_x = self.dAdx_cpu(x)
        AQ_inv_AT = A_x @ torch.linalg.inv(Q) @ A_x.T
        P = -torch.linalg.inv(Q) + torch.linalg.inv(Q)@A_x.T @ torch.linalg.inv(AQ_inv_AT) @ A_x @ torch.linalg.inv(Q)

        # Compute derivatives
        u = P @ lam
        mu = -torch.linalg.inv(AQ_inv_AT) @ A_x @ torch.linalg.inv(Q) @ lam
        u = u.squeeze()

        if mu.shape == (1, 1):
            mu = mu.reshape(1)
        elif mu.shape == (1,):
            mu = mu
        else:
            mu = mu.squeeze()

        dx_dt = u
        dlambda_dt = -torch.einsum('j,jki,k->i', mu, dA_x, u)
        z_dot = torch.cat((dx_dt, dlambda_dt), dim=0)
        return z_dot.to('cpu')

    def A(self, x):
        """Constraint matrix A(x)."""
        theta = x[:,2]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        A_matrix = torch.stack([-sin_theta, cos_theta, torch.zeros_like(theta)], dim=1)
        return A_matrix.unsqueeze(1) 

    def dAdx(self, x):
        """Derivative of A(x) with respect to x for batched inputs."""
        theta = x[:, 2]  # Extract the third column (batched)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        dA_dx = torch.zeros((x.shape[0], 3, 3), device=x.device)  # Shape: (batch_size, 3, 3)
        dA_dx[:, 0, 2] = -cos_theta  # Update batched dA_dx
        dA_dx[:, 1, 2] = -sin_theta
        return dA_dx.unsqueeze(1)  # Shape: (batch_size, 1, 3, 3)


    def ode_system(self, t, y, Q, x_dim):
        """Compute derivatives for batched inputs."""
        batch_size = y.shape[0]
        x = y[:, :x_dim]  # First x_dim columns (x)
        lam = y[:, x_dim:]  # Remaining columns (lambda)

        # Constraint matrix and its derivative (batched)
        A_x = self.A(x)  # Shape: (batch_size, 1, 3)
        dA_x = self.dAdx(x)  # Shape: (batch_size, 1, 3, 3)
        Q_inv = torch.linalg.inv(Q).unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, 3, 3)

        # Compute AQ_inv_AT using batch matrix multiplication
        AQ_inv_AT = torch.bmm(A_x @ Q_inv, A_x.transpose(1, 2))  # Shape: (batch_size, 1, 1)

        # Compute P using batched operations
        P = -Q_inv + torch.bmm(torch.bmm(Q_inv, A_x.transpose(1, 2)), torch.bmm(torch.linalg.inv(AQ_inv_AT),torch.bmm(A_x, Q_inv)))

        u = torch.bmm(P, lam.unsqueeze(2)).squeeze(2)
        mu = torch.bmm(torch.linalg.inv(AQ_inv_AT), torch.bmm(A_x, torch.linalg.inv(Q).unsqueeze(0).expand(batch_size, -1, -1)).bmm(lam.unsqueeze(2))).squeeze(2)
        dlambda_dt = -torch.einsum('bj,bjki,bk->bi', mu, dA_x, u)
        # Concatenate dx/dt and d(lambda)/dt
        z_dot = torch.cat((u, dlambda_dt), dim=1)
        return z_dot

    def lam_to_x_function_cpu(self, lam_batch):
        """
        Process a batch of initial lambda values to compute x(T).

        Parameters:
            lam_batch: torch.Tensor of shape (batch_size, 3) - Initial lambda values.

        Returns:
            xT_batch: torch.Tensor of shape (batch_size, x_dim) - Final x values at time T.
        """
        batch_size = lam_batch.shape[0]
        x0_batch = self.x0.unsqueeze(0).repeat(batch_size, 1)
        x_dim = x0_batch.shape[1]
        xT_batch = []

        for i in range(batch_size):
            # Extract the initial conditions for this batch element
            lam0 = lam_batch[i]
            x0 = x0_batch[i]

            # Combine x0 and lam0
            y0 = torch.cat((x0, lam0), dim=0)

            # Solve the ODE
            sol = solve_ivp(
                self.ode_system_cpu,
                [0, self.T],
                y0.detach().to('cpu'),
                args=(self.Q, x_dim),
                method='RK45',
                t_eval=np.linspace(0, self.T, 100)
            )

            # Extract x(T) and append to the batch result
            xT = torch.tensor(sol.y[:x_dim, -1], dtype=torch.float32, device = device)
            xT_batch.append(xT)

        # Stack the results into a tensor
        xT_batch = torch.stack(xT_batch, dim=0)
        return xT_batch

    def lam_to_x_function(self, lam_batch):

        batch_size = lam_batch.shape[0]
        x0_batch = self.x0.unsqueeze(0).repeat(batch_size, 1).to(device)
        y0_batch = torch.cat((x0_batch, lam_batch), dim=1)  # Combine x0 and lam0

        # Define the ODE function compatible with torchdiffeq
        def ode_func(t, y):
            return self.ode_system(t, y, self.Q, x0_batch.shape[1])  # Pass extra args as needed

        # Solve the ODE for all batch elements in parallel
        t_span = torch.tensor([0, self.T], device=device, dtype=torch.float32)
        yT = odeint(ode_func, y0_batch, t_span, method='rk4')  # Choose appropriate method

        # Extract x(T) (first x_dim elements) for all batch elements
        xT_batch = yT[-1, :, :x0_batch.shape[1]]  # Final time step, all batches, only x_dim
        return xT_batch

    def implement_optimal_trajectory(self, x0, T, Q, lam):
        x_dim = len(x0)
        # Solve the ODE with the optimal lambda(0)
        y0 = torch.cat((x0, lam),dim=0).to('cpu')
        sol = solve_ivp(
            self.ode_system_cpu, [0, T], y0, args=(Q, x_dim),
            method='RK45', t_eval=np.linspace(0, T, 100)
        )
        return sol


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
    plot_path = 'plots/trajectory2D.png'
    plt.savefig(plot_path)
    plt.close()

def clip_to_pi(theta):
    """
    Clips the input angle(s) to the range [-pi, pi].
    
    Parameters:
        theta (torch.Tensor or float): Input angle(s) in radians.
        
    Returns:
        torch.Tensor or float: Angle(s) clipped to the range [-pi, pi].
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


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
    ax.plot(x_vals[:, 0], x_vals[:, 1], clip_to_pi(x_vals[:, 2]), label="Trajectory", color="b")

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Theta')

    # Set title
    ax.set_title('Trajectory on R^2 x S^1 Manifold')

    # Show plot
    plot_path = 'plots/trajectory3D.png'
    plt.savefig(plot_path)
    plt.close()

class CustomLoss(nn.Module):
    def __init__(self, Q):
        """
        Custom loss function for \delta p^T Q \delta p.

        Parameters:
            Q (torch.Tensor): A 3x3 positive-definite matrix.
        """
        super(CustomLoss, self).__init__()
        self.Q = Q

    def forward(self, p1, p2):
        """
        Computes the custom loss.

        Parameters:
            p1 (torch.Tensor): Predicted tensor of shape (batch_size, 3).
            p2 (torch.Tensor): Ground truth tensor of shape (batch_size, 3).

        Returns:
            torch.Tensor: Computed loss.
        """
        delta_p = p1 - p2
        delta_p[:, 2] = clip_to_pi(delta_p[:, 2])  # Clip theta to [-pi, pi]
        # Compute \delta p^T Q \delta p
        loss = torch.matmul(delta_p.unsqueeze(1), torch.matmul(self.Q, delta_p.unsqueeze(2)))
        return loss.mean()  # Return the mean loss over the batch

class XDataset(Dataset):
    def __init__(self, num_samples, bounds):
        """
        Dataset for generating random x vectors normalized to [-1, 1].
        Args:
            num_samples: Number of samples to generate.
            bounds: List of tuples representing the bounds for x1, x2, x3.
        """
        self.num_samples = num_samples
        self.bounds = bounds

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly sample x values within the specified bounds
        x1 = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
        x2 = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
        x3 = np.random.uniform(self.bounds[2][0], self.bounds[2][1])
        
        # Normalize each value to [-1, 1]
        x1 = 2 * (x1 - self.bounds[0][0]) / (self.bounds[0][1] - self.bounds[0][0]) - 1
        x2 = 2 * (x2 - self.bounds[1][0]) / (self.bounds[1][1] - self.bounds[1][0]) - 1
        x3 = 2 * (x3 - self.bounds[2][0]) / (self.bounds[2][1] - self.bounds[2][0]) - 1
        
        x = torch.tensor([x1, x2, x3], dtype=torch.float32, device=device)
        
        return x

def de_normalize(x, bounds):
    """
    De-normalize a batch of tensors in GPU.
    Args:
        x: Tensor of shape (batch_size, dim) normalized in range [-1, 1].
        bounds: List of tuples [(min1, max1), (min2, max2), ...] defining bounds for each dimension.
    Returns:
        De-normalized tensor of shape (batch_size, dim).
    """
    bounds = torch.tensor(bounds, dtype=torch.float32, device=x.device)  # Convert bounds to tensor on the same device
    mins = bounds[:, 0]  # Extract minimum values
    maxs = bounds[:, 1]  # Extract maximum values

    # De-normalize in a single vectorized operation
    return (x + 1) * (maxs - mins) / 2 + mins


def main(mode):
    bounds = [(-2, 2), (-2, 2), (-np.pi, np.pi)]
    if mode == 'train': 
        system = ODESystem()
        num_samples = 102400 # Number of training samples
        batch_size = 128
        num_epochs = 1000000
        learning_rate = 1e-4
        
        # Dataset and DataLoader
        dataset = XDataset(num_samples=num_samples, bounds=bounds)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Model, optimizer, and loss function
        model = XtolamNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = CustomLoss(system.Q)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)  


        val_num_samples = 1000
        val_dataset = XDataset(num_samples=val_num_samples, bounds=bounds)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')  # Initialize with a high value

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for x_batch in dataloader:
                x_batch = x_batch.to(device)
                
                # Forward pass: Get the model's prediction
                predicted_lam = model(x_batch)
                predicted_x = system.lam_to_x_function(predicted_lam)
                predicted_x[:,2] = clip_to_pi(predicted_x[:,2])
                # Compute the loss
                x_orig = de_normalize(x_batch,bounds)
                loss = criterion(predicted_x, x_orig)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Predicted X: {predicted_x[0]}")
            print(f"Actual X: {x_orig[0]}")
            
            # Validation phase
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # No gradients needed for validation
                for x_batch in val_dataloader:
                    x_batch = x_batch.to(device)
                    
                    # Forward pass: Get the model's prediction
                    predicted_lam = model(x_batch)
                    predicted_x = system.lam_to_x_function(predicted_lam)
                    predicted_x[:,2] = clip_to_pi(predicted_x[:,2])
                    x_orig = de_normalize(x_batch, bounds)
                    # Compute the loss
                    val_loss += criterion(predicted_x, x_orig).item()
            
            # Calculate average validation loss
            val_loss /= len(val_dataloader)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(dataloader)}, Validation Loss: {val_loss}")

            # Save the model weights if the validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save model weights
                if epoch > 50:
                    torch.save(model.state_dict(), f'weight/best_model_epoch_{epoch+1}.pth')
                    print(f"Best validation loss improved! Model saved at epoch {epoch+1}")

    elif mode == 'val':
        # Validation stage
        system = ODESystem()
        x0 = system.x0
        x0 = x0.to(device)
        T = system.T
        Q = system.Q
        Q = Q.to(device)
        x_dim = len(x0)
        model = XtolamNet().to(device)
        best_model_path = "weight/best_model_epoch_1627.pth"
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # Example validation input
        test_sample = torch.tensor([[1.0, 2.0, 0.7]], device=device)
        bounds = torch.tensor(bounds, dtype=torch.float32, device=device)
        test_sample_input = 2 * (test_sample - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1
        with torch.no_grad():
            prediction = model(test_sample_input).to(device)
            test_sample = test_sample.to(device)
            print("Validation Input x(T):", test_sample)
            print("Validation Predicted lambda(0):", prediction)
        
        x_T = test_sample
        lam = prediction.reshape(3) 
        solution = system.implement_optimal_trajectory(x0, T, Q, lam)
        tensor1 = x_T
        tensor2 = solution.y[:x_dim,-1]
        print(f"Real endpoint: {tensor1.tolist()} Estimated endpoint: {tensor2.tolist()}")
        t_vals = solution.t
        x_vals = solution.y[:len(x0), :].T
        lam_vals = solution.y[len(x0):, :].T
        visualize_trajectory(t_vals, x_vals)
        vis_3d(t_vals, x_vals)

    
    else:
        raise ValueError("Invalid mode! Use 'train', or 'val'.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train, validate, or generate data.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'val'],
                        help="Mode to run the script: 'train', or 'val'.")
    args = parser.parse_args()

    # Run the main function with the specified mode
    main(mode=args.mode)