import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, dt=0.1, sp=0.001, sm=0.05, use_lag_smoother=False):
        self.dt = dt  # time step
        self.sp = sp  # process noise parameter
        self.sm = sm  # measurement noise parameter
        self.use_lag_smoother = use_lag_smoother

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


        # _____Lag smoother parameters_____

        self.N = 5  # lag size
        self.dim = 6  # state dimension
        self.aug_dim = self.dim * (self.N + 1)  # augmented state dimension

        self.x_aug = np.zeros((self.aug_dim, 1))
        self.x_aug[0:6] = np.array([-10, -150, 1, -2, 0, 0], dtype=float).reshape(-1, 1)

        # Augmented state transition matrix
        self.psi_aug = np.zeros((self.aug_dim, self.aug_dim))
        self.psi_aug[0:6, 0:6] = self.psi
        for i in range(1, self.N + 1):
            self.psi_aug[i*6 : (i+1)*6, (i-1)*6 : i*6] = np.eye(6)

        # Augmented measurement matrix
        self.phi_aug = np.zeros((2, self.aug_dim))
        self.phi_aug[:, 0:6] = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        self.P_aug = np.eye(self.aug_dim)  # initial augmented covariance matrix
        self.S = self.sm * np.eye(2)  # measurement noise covariance for augmented state

        self.Q_aug = np.zeros((self.aug_dim, self.aug_dim))
        self.Q_aug[0:6, 0:6] = self.sp * np.eye(6)

    def predict(self):
        if not self.use_lag_smoother:
            self.x = self.psi @ self.x
            self.P = self.psi @ self.P @ self.psi.T + self.sp * np.eye(6)

            return self.x, self.P
        else:
            self.x_aug = self.psi_aug @ self.x_aug
            self.P_aug = self.psi_aug @ self.P_aug @ self.psi_aug.T + self.Q_aug
            return self.x_aug[0:6], self.P_aug
    
    def update(self, z):
        
        if not self.use_lag_smoother:
            z = z.reshape(-1, 1)

            # Calculate Kalman Gain
            K = self.P @ self.phi.T @ np.linalg.inv(self.S + (self.phi @ self.P @ self.phi.T))

            # State update
            self.x = self.x + K @ (z - self.phi @ self.x)

            # Covariance update
            self.P = (np.eye(6) - K @ self.phi) @ self.P

            return self.x, self.P
        
        else:
            z = z.reshape(-1, 1)

            # Calculate Kalman Gain
            K = self.P_aug @ self.phi_aug.T @ np.linalg.pinv(self.S + (self.phi_aug @ self.P_aug @ self.phi_aug.T))

            # State update
            self.x_aug = self.x_aug + K @ (z - self.phi_aug @ self.x_aug)

            # Covariance update
            self.P_aug = (np.eye(self.aug_dim) - K @ self.phi_aug) @ self.P_aug

            return self.x_aug, self.P_aug

    def run_and_visualize(self, observations):
        num_steps = observations.shape[0]
        state_history = np.zeros((num_steps, 6))

        if not self.use_lag_smoother:
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

        else:
            for t in range(num_steps):
                self.predict()

                current_obs = observations[t]
                if not np.isnan(current_obs).any():
                    self.update(current_obs)
                else:
                    pass
                
                state_history[t] = self.x_aug[-6:].flatten()


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
    # this is the kf implementation and visualization
    kf = KalmanFilter(dt=0.1, sp=0.001, sm=0.05, use_lag_smoother=False)
    kf.run_and_visualize(observations)

    # TASK 2
    # run fixed lag smoother with lag of 5 steps
    kf = KalmanFilter(dt=0.1, sp=0.001, sm=0.05, use_lag_smoother=True)
    kf.run_and_visualize(observations)

    # TASK 3
    # analysis of effects of varying lag size

    pass

if __name__ == "__main__":
    main()
