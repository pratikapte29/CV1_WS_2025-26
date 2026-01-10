import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, dt=0.1, sp=0.001, sm=0.05):
        self.dt = dt  # time step
        self.sp = sp  # process noise parameter
        self.sm = sm  # measurement noise parameter

        self.x = np.array([-10, -150, 1, -2, 0, 0], dtype=float).reshape(-1, 1)  # initial state vector

        self.psi = np.array(
            [[1, 0, dt, 0, 0.5 * dt**2, 0],
             [0, 1, 0, dt, 0, 0.5 * dt**2],
             [0, 0, 1, 0, dt, 0],
             [0, 0, 0, 1, 0, dt],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]], dtype=float
        )  # state transition matrix

        self.phi = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ]
        )  # measurement matrix

        self.P = np.eye(6) # * 1000.0  # initial covariance matrix
        self.S = self.sm * np.eye(2)  # measurement noise covariance

    def predict(self):
        self.x = self.psi @ self.x
        self.P = self.psi @ self.P @ self.psi.T + self.sp * np.eye(6)

        return self.x, self.P
    
    def update(self, z):

        z = z.reshape(-1, 1)

        # Calculate Kalman Gain
        K = self.P @ self.phi.T @ np.linalg.inv(self.S + (self.phi @ self.P @ self.phi.T))

        # State update
        self.x = self.x + K @ (z - self.phi @ self.x)

        # Covariance update
        self.P = (np.eye(6) - K @ self.phi) @ self.P

        return self.x, self.P
    
    def run_and_visualize(self, observations):
        num_steps = observations.shape[0]
        state_history = np.zeros((num_steps, 6))

        for t in range(num_steps):
            self.predict()

            # Check if the observation is valid (not NaN)
            current_obs = observations[t]
            if not np.isnan(current_obs).any():
                # Only update if we have a valid measurement
                self.update(current_obs)
            else:
                pass
            
            state_history[t] = self.x.flatten()

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.plot(observations[:, 0], observations[:, 1], 'ro', label='Observations')
        plt.plot(state_history[:, 0], state_history[:, 1], 'b-', label='Kalman Filter Estimate')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Kalman Filter Tracking')
        plt.legend()
        plt.grid()
        plt.show()


def main():
    # Load observations # 1
    observations = np.load('data/observations.npy')
 
    # TASK 1
    kf = KalmanFilter(dt=0.1, sp=0.001, sm=0.05)
    kf.run_and_visualize(observations)

    # TASK 2

    # TASK 3

    pass

if __name__ == "__main__":
    main()
