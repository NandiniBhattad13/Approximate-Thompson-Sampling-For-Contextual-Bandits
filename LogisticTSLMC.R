# Sigmoid and normalize helper functions
sigmoid <- function(u) 
{
  1 / (1 + exp(-u))
}

normalize <- function(u) 
{
  denom <- sqrt(sum(u^2))
  if (denom > 0) 
  {
    return(u / denom)
  } 
  else 
  {
    return(u)
  }
}

# Gradient of the negative log-posterior for the logistic model
grad_neg_log_posterior <- function(theta, X_hist, r_hist, n_obs, sigma0) 
{
  # Gradient from the Gaussian prior
  grad_prior <- theta / (sigma0^2)
  
  # Gradient from the likelihood (if we have observations)
  if (n_obs > 0) 
  {
    X_obs <- X_hist[1:n_obs, , drop = FALSE]
    r_obs <- r_hist[1:n_obs]
    
    pred_probs <- sigmoid(X_obs %*% theta)
    
    # Gradient of negative log-likelihood is -sum( (r - p) * x )
    grad_likelihood <- -crossprod(X_obs, (r_obs - pred_probs))
    return(as.vector(grad_prior + grad_likelihood))
  } 
  else 
  {
    return(as.vector(grad_prior))
  }
}


# Parameters
T <- 10000
K <- 50
d_context <- 20
sigma0 <- 0.01

# LMC hyperparameters
h <- 1e-5      # Step size 
beta <- 1.0     # Inverse temperature
K_lmc <- 20     # Number of LMC steps per round

set.seed(123)

# True parameter vector
theta_star <- rnorm(d_context, mean = 0, sd = 1)
theta_star <- normalize(theta_star)

# theta_curr is the current state of our LMC sampler
theta_curr <- rep(0, d_context)

X_history <- matrix(0, nrow = T, ncol = d_context)
r_history <- numeric(T)
t_obs <- 0


regrets <- numeric(T)
chosen_arms <- numeric(T)
optimal_arms <- numeric(T)


for (t in 1:T) 
{
  # 1. Generate new contexts for all arms
  arm_contexts <- matrix(rnorm(n = K * d_context, mean = 0, sd = 1), nrow = K, ncol = d_context)
  arm_contexts <- t(apply(arm_contexts, 1, normalize))
  
  # 2. Sample theta from posterior using LMC
  for (k in 1:K_lmc) 
  {
    noise <- rnorm(d_context)
    grad <- grad_neg_log_posterior(theta_curr, X_history, r_history, t_obs, sigma0)
    theta_curr <- theta_curr - h * grad + sqrt(2 * h / beta) * noise
  }
  
  # 3. Action Selection: Choose arm based on the sampled theta
  pred_probs <- sigmoid(arm_contexts %*% theta_curr)
  a_t <- which.max(pred_probs)
  chosen_arms[t] <- a_t
  
  # 4. Environment: Generate reward for the chosen arm
  chosen_context <- arm_contexts[a_t, ]
  true_prob <- sigmoid(sum(chosen_context * theta_star))
  r_t <- rbinom(1, 1, true_prob)
  
  # 5. Update History
  t_obs <- t_obs + 1
  X_history[t_obs, ] <- chosen_context
  r_history[t_obs] <- r_t
  
  # 6. Calculate Regret
  true_all_probs <- sigmoid(arm_contexts %*% theta_star)
  optimal_reward <- max(true_all_probs)
  optimal_arms[t] <- which.max(true_all_probs)
  chosen_arm_reward <- true_all_probs[a_t]
  regrets[t] <- optimal_reward - chosen_arm_reward
}


cumulative_regret <- cumsum(regrets)
cat(sprintf("\nFinal cumulative regret: %.2f\n", cumulative_regret[T]))

# Accuracy of arm selection
accuracy <- mean(chosen_arms == optimal_arms)
cat(sprintf("\nAccuracy of optimal arm selection: %.2f%%\n", 100 * accuracy))

# Plot cumulative regret
plot(cumulative_regret, type = "l", col = "red", lwd = 2,
     main = "Cumulative Regret for LMC Logistic Bandit",
     xlab = "Round (t)", ylab = "Cumulative Regret")

# Plot the fraction of times the optimal arm was chosen
optimal_fraction <- cumsum(chosen_arms == optimal_arms) / (1:T)
plot(optimal_fraction, type = "l", col = "blue", lwd = 2,
     ylim = c(0, 1),
     main = "Fraction of Optimal Arm Selections (LMC)",
     xlab = "Round (t)", ylab = "Fraction Optimal")
