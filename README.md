📊 Bayesian vs. Frequentist Statistics: A Visual Comparison

This project demonstrates the differences between Bayesian and Frequentist statistics using two examples:
- **Coin Flip Simulation**: How do both methods estimate the probability of "heads"?
- **Doping Test Simulation**: Why a positive test does not necessarily mean an athlete is doping.

What’s the Difference?

**Frequentist Statistics**
- Treats probabilities as fixed values.
- Assumes probabilities can only be determined through infinite experimentation.
- Example: If we flip a coin 100 times, the estimated probability of heads is simply;

P = Number of Heads / Number of coin flips

**Bayesian Statistics**  
- Treats probabilities as subjective beliefs that are updated with new data.
- Uses **Bayes’ Theorem**:  

P(H | D) = (P(D | H) * P(H) / P(D) 

  - **P(H | D)**: Probability of the hypothesis given the observed data (**Posterior**)
  - **P(D | H)**: Probability of the data given the hypothesis (**Likelihood**)
  - **P(H)**: Prior belief about the hypothesis (**Prior**)
  - **P(D)**: Total probability of the data (**Normalization Factor**)

1. **Coin Flip Simulation** 
**Goal:** Estimate the probability that a coin lands on heads.

- **Frequentist:** Calculates the mean
- **Bayesian:** Starts with an assumption (Prior) about the probability and updates it with new flips.

**Code:**

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Simulating a coin flip with a "true" probability
true_p = 0.5
n_trials = 100
np.random.seed(42)  # For reproducibility
coin_flips = np.random.binomial(1, true_p, n_trials)

# Frequentist estimate
freq_estimate = np.sum(coin_flips) / n_trials

# Bayesian estimate with Beta(1,1) prior
alpha_prior= 1
beta_prior = 1
alpha_post = alpha_prior + np.sum(coin_flips)
beta_post = beta_prior + n_trials - np.sum(coin_flips)

# Plot posterior distribution
x = np.linspace(0, 1, 100)
plt.figure(figsize=(10,5))
plt.axvline(freq_estimate, color='red', linestyle='--', label=f"Frequentist: p = {freq_estimate:.2f}")
plt.plot(x, stats.beta.pdf(x, alpha_post, beta_post), label="Bayesian Posterior", color='blue')
plt.xlabel("Probability of Heads")
plt.ylabel("Density")
plt.title("Frequentist vs Bayesian Coin Toss Estimation")
plt.legend()
plt.show()

![image](https://github.com/user-attachments/assets/d050e88d-02c0-4fda-bc43-8c93b2e5d3c0)



2. **Doping Test Simulation**
**Goal:** Calculate the actual probability of doping after a positive test result.

- **Frequentist:** Only considers the test sensitivity (98%).
- **Bayesian:** Takes into account that only 0.5% of athletes dope and combines this with the test result.

**Code: `scripts/doping_test.py`**
import numpy as np
import matplotlib.pyplot as plt

# Given probabilities
P_doping = 0.005  # Only 0.5% of athletes dope
P_pos_given_doping = 0.98  # Test correctly detects doping 98% of the time
P_pos_given_clean = 0.05  # 5% false positives

# Bayesian calculation
P_clean = 1 - P_doping
P_pos = P_pos_given_doping * P_doping + P_pos_given_clean * P_clean
P_doping_given_pos = (P_pos_given_doping * P_doping) / P_pos

# Visualization
labels = ["Frequentist: Test Sensitivity", "Bayesian: True Doping Probability"]
values = [P_pos_given_doping, P_doping_given_pos]

plt.figure(figsize=(8,5))
plt.bar(labels, values, color=["red", "blue"])
plt.ylabel("Probability")
plt.title("Frequentist vs Bayesian Doping Probability")
plt.ylim(0, 1)
plt.show()

print(f"Frequentist Estimate: {P_pos_given_doping:.2%} (Test Sensitivity)")
print(f"Bayesian Estimate: {P_doping_given_pos:.2%} (Actual Doping Probability after a Positive Test)")

![image](https://github.com/user-attachments/assets/a712bef9-77ea-4c77-a6f9-57b5b5cd9865)
