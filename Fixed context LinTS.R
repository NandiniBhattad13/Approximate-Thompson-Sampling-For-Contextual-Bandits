library(MASS)

# Parameters
T <- 10000
K <- 5
d_context <- 4
d <- K * d_context
sigma <- 0.5    # reward noise
sigma0 <- 0.01  # prior std dev

set.seed(123)

# Fixed context (same for all rounds)
X_fixed <- rnorm(d_context, mean = 0, sd = 1)

# True parameter vector
theta_star <- rnorm(d, mean = 0, sd = sigma0)

# Feature map 
phi <- function(X, i) 
{
  vec <- rep(0, d)
  start_idx <- (i - 1) * d_context + 1
  end_idx <- start_idx + d_context - 1
  vec[start_idx:end_idx] <- X
  return(vec)
}

# compute true expected reward of each arm
true_rewards <- sapply(1:K, function(a) sum(phi(X_fixed, a) * theta_star))
# find best arm
best_arm <- which.max(true_rewards)


# Prior mean and covariance
mu <- rep(0, d)
Sigma <- diag(sigma0^2, d)

# Storage
regrets <- numeric(T)
chosen_arms <- numeric(T)

for (t in 1:T) 
{
  # Thompson sample
  theta_sample <- MASS::mvrnorm(1, mu, Sigma)
  
  # Predicted rewards under sampled theta
  pred_rewards <- sapply(1:K, function(a) sum(phi(X_fixed, a) * theta_sample))
  a_t <- which.max(pred_rewards)
  chosen_arms[t] <- a_t
  
  # Observe noisy reward
  eps_t <- rnorm(1, mean = 0, sd = sigma)
  r_t <- sum(phi(X_fixed, a_t) * theta_star) + eps_t
  
  # Compute regret (expected, noise-free)
  opt_rewards <- sapply(1:K, function(a) sum(phi(X_fixed, a) * theta_star))
  r_star_t <- max(opt_rewards)
  regrets[t] <- r_star_t - sum(phi(X_fixed, a_t) * theta_star)
  
  # Posterior update
  phi_vec <- phi(X_fixed, a_t)
  Sigma_inv <- solve(Sigma)
  Sigma <- solve(Sigma_inv + (1 / sigma^2) * (phi_vec %*% t(phi_vec)))
  mu <- Sigma %*% (Sigma_inv %*% mu + (1 / sigma^2) * phi_vec * r_t)
}

# Cumulative regret
cumulative_regret <- cumsum(regrets)
cat(sprintf("Final cumulative regret: %.2f\n", cumulative_regret[T]))
cat("\nArm selection frequencies:\n")
print(table(chosen_arms))
barplot(table(chosen_arms), col="purple", 
        main="Histogram of Chosen Arms", 
        xlab="Arm", ylab="Number of times selected")
