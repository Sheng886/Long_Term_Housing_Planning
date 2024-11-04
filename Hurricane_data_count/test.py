import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Step 1: Generate Training Data
np.random.seed(42)
# Assume the true parameter is lambda = 4
true_lambda = 4
# Generate 100 observations from a Poisson distribution with lambda = 4
observed_data = np.random.poisson(lam=true_lambda, size=100)

# Step 2: Define the Log-Likelihood Function
def log_likelihood(lambda_param, data):
    # Poisson log-likelihood
    if lambda_param <= 0:
        return -np.inf  # To ensure lambda is positive
    return np.sum(stats.poisson.logpmf(data, mu=lambda_param))

# Step 3: Define the Monte Carlo Maximum Likelihood Estimation (MCMLE) Function with Metropolis-Hastings and Adaptive Step Size
def mcmle(data, initial_lambda, n_iterations=1000):
    # Initial parameter guess
    best_lambda = initial_lambda
    best_log_likelihood = log_likelihood(best_lambda, data)
    
    # Step size for candidate generation
    step_size = 0.5
    accept_count = 0
    
    # Monte Carlo iterations to find the best lambda
    for iteration in range(n_iterations):
        # Generate a candidate lambda by adding Gaussian noise to the current best lambda
        candidate_lambda = best_lambda + np.random.normal(0, step_size)
        # Ensure candidate lambda is positive
        candidate_lambda = max(candidate_lambda, 0.01)
        
        # Calculate log likelihood for the candidate lambda
        candidate_log_likelihood = log_likelihood(candidate_lambda, data)
        
        # Metropolis-Hastings acceptance criterion
        acceptance_probability = min(1, np.exp(candidate_log_likelihood - best_log_likelihood))
        if np.random.rand() < acceptance_probability:
            best_lambda = candidate_lambda
            best_log_likelihood = candidate_log_likelihood
            accept_count += 1
        
        # Adaptive step size adjustment every 100 iterations
        if (iteration + 1) % 100 == 0:
            acceptance_rate = accept_count / 100
            if acceptance_rate > 0.5:
                step_size *= 1.1  # Increase step size if acceptance rate is high
            elif acceptance_rate < 0.2:
                step_size *= 0.9  # Decrease step size if acceptance rate is low
            accept_count = 0
            
    return best_lambda

# Step 4: Run MCMLE to Estimate Lambda
initial_guess = 2.0  # Initial guess for lambda
estimated_lambda = mcmle(observed_data, initial_guess)

print(f"Estimated Lambda using MCMLE: {estimated_lambda}")
print(f"True Lambda: {true_lambda}")

# Step 5: Plot Observed Data vs. Model Prediction
plt.hist(observed_data, bins=10, alpha=0.5, label='Observed Data', color='blue')
plt.axvline(estimated_lambda, color='red', linestyle='dashed', linewidth=2, label=f'Estimated Lambda = {estimated_lambda:.2f}')
plt.axvline(true_lambda, color='green', linestyle='dotted', linewidth=2, label=f'True Lambda = {true_lambda}')
plt.xlabel('Count Value')
plt.ylabel('Frequency')
plt.title('Observed Data vs. Estimated Lambda')
plt.legend()
plt.show()
