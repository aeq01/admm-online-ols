import numpy as np
from numpy.linalg import inv


class ADMMBlockRegression:
    """
    Class is used to store intermediate representations of the data.
    Allows for streaming new data in and performing real-time updates of linear regression model.
    """

    def __init__(self, lambda_: float, p: float, x_dimension: int) -> None:
        # data that we cache for speeding each iteration
        self.lhs_matrices: [np.array] = []
        self.rhs_matrices: [np.array] = []

        # general ADMM constants
        self.lambda_: float = lambda_
        self.p: int = p
        self.n_dim_identity_p: np.array = np.diag([p] * (x_dimension))
        self.x_dimension: int = x_dimension

        # ADMM variables
        self.current_x: [np.array] = []
        self.current_u: [np.array] = []
        self.current_z: np.array = np.zeros(x_dimension)  # consensus variable

    def insert_block(self, data_block, output_block) -> None:
        """
        Processes the new data and inserts it into internal data structures

        Args:
            data_block: n rows and x_dimension columns. Represents new data to insert
            output_block: n rows and 1 column. Represents the output for the new data

        """
        # process each raw data matrix and cache the relevant results
        self.lhs_matrices.append(inv(data_block.T @ data_block + self.n_dim_identity_p))
        self.rhs_matrices.append(output_block @ data_block)

        # add in another spot in x and u for the new block
        self.current_x.append(np.zeros(self.x_dimension))
        self.current_u.append(np.zeros(self.x_dimension))

    def run_iteration(self) -> None:
        """
        Runs a single ADMM update step.
        Algorithm given in Section 8.2: https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
        """

        for j in range(len(self.current_x)):
            self.current_x[j] = self._update_x(j)

        self.current_z = self._update_z()

        for j in range(len(self.current_x)):
            self.current_u[j] = self._update_u(j)

    def get_fitted_slope(self) -> np.array:
        """
        Returns the final slope that the consensus variable stores
        """
        return self.current_z

    def _update_x(self, index) -> np.array:
        """
        Updates x_index with
        Algorithm given in Section 8.2: https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

        Args:
            index: we have x_1, x_2...x_N. This function updates x_index.
        """
        rhs_vector = self.rhs_matrices[index] + self.p * (
            self.current_z - self.current_u[index]
        )
        return self.lhs_matrices[index] @ rhs_vector

    def _update_z(self) -> np.array:
        """
        Updates z with
        Algorithm given in Section 8.2: https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
        Forces each x_1.. x_i to agree on one result (current_z)
        """
        x_k: np.array = sum(self.current_x) / len(self.current_x)
        u_k: np.array = sum(self.current_u) / len(self.current_u)
        threshold_: float = self.lambda_ / (self.p * (len(self.current_x)))

        def soft_threshold(alpha_: float, threshold_: float) -> float:
            if alpha_ > threshold_:
                return alpha_ - threshold_
            elif alpha_ < (-1 * threshold_):
                return alpha_ + threshold_
            else:
                return 0

        new_vec = x_k + u_k

        # vectorize the func so it can easily be applied to each element of new_vec
        vfunc = np.vectorize(soft_threshold)
        return vfunc(new_vec, threshold_)

    def _update_u(self, index) -> np.array:
        """
        Updates u_index with
        Algorithm given in Section 8.2: https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

        Args:
            index: we have u_1, u_2...u_N. This function updates u_index.
        """
        return self.current_u[index] + self.current_x[index] - self.current_z
