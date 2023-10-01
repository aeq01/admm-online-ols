from sklearn import linear_model
import time
import numpy as np


class SKLearnLinearRegressionWrapper:
    def fit_and_find_coef(x_data, y_data, x_dim):
        """
        Takes sample input/output data and uses sklearn to fit a linear model

        Args:
            x_data: n rows by x_dim columns, data to fit the model on
            y_data: n rows by 1 column, output data to fit the model on
            x_dim: number of dimensions for the input in our model

        Returns:
            fit_time: amount of time to fit the model
            slope: resulting relevant coefficients fo the model
        """
        clf = linear_model.LinearRegression()

        start_lin_reg = time.time()
        clf.fit(x_data, y_data)
        end_lin_reg = time.time()
        fit_time = end_lin_reg - start_lin_reg

        # start with just the intercept
        intercept = clf.predict(np.array([[0] * (x_dim + 1)]))

        # used to produce slopes
        coefs = [intercept]

        # we skip [1,0,...0] because typically fits to 0. [0,0...0] gives us the intercept
        # we pass [0,1, 0...0], [0, 0, 1,0....0] through the model to determine the coefs
        for i in range(1, x_dim + 1):
            current_input = [0] * (x_dim + 1)
            current_input[i] = 1
            coefs.append(clf.predict(np.array([current_input])) - intercept)

        slope = np.array(coefs).reshape((x_dim + 1))
        return fit_time, slope
