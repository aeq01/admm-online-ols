import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import time
import pdb
from data_generator.Data_Manager import NoisyFunctionMapper
from data_generator.Data_Manager import DataBlocksManager
from solvers.ADMM_Regression import ADMMBlockRegression
from solvers.SKLearn_Linear_Regression_Wrapper import SKLearnLinearRegressionWrapper

"""
Start with Data Generation
"""

# equations that we pass our random input through to produce the data blocks
equations = [
    NoisyFunctionMapper.equation_1,
    NoisyFunctionMapper.equation_2,
    NoisyFunctionMapper.equation_3,
    NoisyFunctionMapper.equation_2,
    NoisyFunctionMapper.equation_1,
    NoisyFunctionMapper.equation_1,
    NoisyFunctionMapper.equation_2,
    NoisyFunctionMapper.equation_3,
    NoisyFunctionMapper.equation_2,
    NoisyFunctionMapper.equation_1,
    NoisyFunctionMapper.equation_1,
    NoisyFunctionMapper.equation_2,
    NoisyFunctionMapper.equation_3,
    NoisyFunctionMapper.equation_2,
    NoisyFunctionMapper.equation_1,
    NoisyFunctionMapper.equation_2,
    NoisyFunctionMapper.equation_3,
    NoisyFunctionMapper.equation_2,
    NoisyFunctionMapper.equation_1,
    NoisyFunctionMapper.equation_1,
]

# number of data blocks (X_i, y_i) to generate
data_blocks_count = len(equations)
raw_data_blocks = []
raw_output_blocks = []

x_inputs = 4

# size of each (X_i, y_i) block
samples_per_batch = 1000000
total_batches = len(equations)
print(
    f"Samples per batch: {samples_per_batch}, with {total_batches} batches. Max data size: {samples_per_batch * total_batches}"
)
print(f"Number of feature: {x_inputs}")
noisy_function_mapper = NoisyFunctionMapper(x_inputs)

for equation in equations:
    current_X_data, current_y_data = noisy_function_mapper.produce_data(
        samples_per_batch, equation
    )
    raw_data_blocks.append(np.copy(current_X_data))
    raw_output_blocks.append(np.copy(current_y_data))

# DataBlocksManager does some pre-processing to slice the data so that it is available in different formats
data_blocks = DataBlocksManager(raw_data_blocks)
output_blocks = DataBlocksManager(raw_output_blocks)

"""
We now fit the scikit-learn linear regression model
scikit-learn uses data_blocks.blocks and output_blocks.blocks because we need the full datasets
"""

lin_reg_fit_times = []
coef_fits = []
for i in range(data_blocks_count):
    fit_time, current_slope_coefs = SKLearnLinearRegressionWrapper.fit_and_find_coef(
        data_blocks.blocks[i], output_blocks.blocks[i], x_inputs
    )
    coef_fits.append(current_slope_coefs)
    lin_reg_fit_times.append(fit_time)


"""
We now fit the naive linear regression model; inv(X.T @ X) @ X.T @ y

Let X_3 be the full block with first two batches. let x_1 and x_2 be the individual batches.
We know X_3.T @ X_3 = x_1.T @ x_1 + x_2.T @ x_2
"""

naive_lin_reg_fit_times = []
naive_coefs = []
lhs_products = []

for i in range(data_blocks_count):
    start_time = time.time()
    if len(lhs_products) == 0:
        # we use naive calculation; slope = inv(X.T @ X) @ X.T @ y
        X, y = data_blocks.blocks[i], output_blocks.blocks[i]
        matrix_product = X.T @ X
        slope = inv(matrix_product) @ X.T @ y  # closed form solution for OLS
        lhs_products.append(matrix_product)
    else:
        # we are able to use cached responses from previous results
        # if X is created by stacking X_1 on top of X_2, then X.T @ X  = X_1.T @ X_1 + X_2.T @ X_2
        X_batch, X_full, y = (
            data_blocks.raw_blocks[i],
            data_blocks.blocks[i],
            output_blocks.blocks[i],
        )
        matrix_product = lhs_products[-1] + X_batch.T @ X_batch
        slope = inv(matrix_product) @ X_full.T @ y  # closed form solution for OLS
        lhs_products.append(matrix_product)
    end_time = time.time()
    naive_coefs.append(slope)
    naive_lin_reg_fit_times.append(end_time - start_time)

"""
We now fit the ADMM linear regression model
ADMM uses data_blocks.raw_blocks and output_blocks.raw_blocks because we can fit N different models on each of the smaller blocks
"""

# ADMM constants
lambda_ = 0  # no regularization currently
p = 10
x_dimension = x_inputs + 1  # 1 extra for intercept
admmRegression = ADMMBlockRegression(lambda_, p, x_dimension)
admm_fit_times = []

for i in range(data_blocks_count):
    start_admm_fit = time.time()

    # at this point we have seen everything up to the i_th data block
    # insert_block does some pre-processing and stores smaller, transformed versions (we don't need full block)
    admmRegression.insert_block(
        data_blocks.raw_blocks[i], output_blocks.raw_blocks[i].T
    )

    # we run 5 ADMM iterations
    for _ in range(5):
        admmRegression.run_iteration()

    end_admm_fit = time.time()
    admm_fit_times.append(end_admm_fit - start_admm_fit)

    fitted_slope = admmRegression.get_fitted_slope()
    admm_vs_optimal_error = LA.norm(fitted_slope - naive_coefs[i])
    print(f"admm_vs_optimal_error: {admm_vs_optimal_error}")

"""
Printing and comparing the solving times for all 3
"""
for i in range(data_blocks_count):
    print(
        f"scikit-learn: {lin_reg_fit_times[i]:.4f} ; admm: {admm_fit_times[i]:.4f} ; naive: {naive_lin_reg_fit_times[i]:.4f}"
    )
