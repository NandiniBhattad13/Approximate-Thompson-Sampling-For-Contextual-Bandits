library(MASS)

# Parameters
T <- 10000     
K <- 5
d_context <- 4
d <- K * d_context
sigma <- 0.5    # reward noise
sigma0 <- 0.01  # prior std dev

# LMC hyperparameters
h <- 1e-6
beta <- 1.0
K_lmc <- 20

set.seed(123)

# True parameter vector
theta_star <- rnorm(d, mean = 0, sd = sigma0)

# Feature map
phi <- function(X, i) 
{
  vec <- rep(0, d)
  start_idx <- (i - 1) * d_context + 1
  vec[start_idx:(start_idx + d_context - 1)] <- X
  return(vec)
}

# Gradient 
grad_loss <- function(theta, A, b, sigma, sigma0) 
{
  grad <- theta / (sigma0^2)
  if (sum(A) > 0)
  {
    grad <- grad + (A %*% theta - b) / (sigma^2)
  }
  return(as.vector(grad))
}

# Tracking
regrets <- numeric(T)
chosen_arms <- numeric(T)
optimal_arms <- numeric(T)
errors1 <- numeric(T)

# Initialize theta
theta_curr <- rep(0, d)

# Sufficient statistics for gradient
A <- matrix(0, d, d)   # Σ φ φ^T
b <- rep(0, d)         # Σ r φ

for (t in 1:T) 
{
  X_t <- rnorm(d_context, mean = 0, sd = 1)
  
  # LMC updates
  for (k in 1:K_lmc) 
  {
    noise <- rnorm(d)
    theta_curr <- theta_curr - h * grad_loss(theta_curr, A, b, sigma, sigma0) + sqrt(2 * h / beta) * noise
  }
  
  # Action selection
  Phi_mat <- sapply(1:K, function(a) phi(X_t, a))
  pred_rewards <- as.vector(crossprod(Phi_mat, theta_curr))
  a_t <- which.max(pred_rewards)
  chosen_arms[t] <- a_t
  
  # Reward
  eps_t <- rnorm(1, sd = sigma)
  r_t <- sum(phi(X_t, a_t) * theta_star) + eps_t
  
  # Optimal reward
  opt_rewards <- as.vector(crossprod(Phi_mat, theta_star))
  opt_arm <- which.max(opt_rewards)
  optimal_arms[t] <- opt_arm
  r_star_t <- max(opt_rewards)
  
  regrets[t] <- r_star_t - sum(phi(X_t, a_t) * theta_star)
  
  # Update sufficient statistics
  phi_t <- phi(X_t, a_t)
  A <- A + tcrossprod(phi_t)
  b <- b + r_t * phi_t
  
  # Error tracking
  errors1[t] <- sqrt(sum((theta_curr - theta_star)^2))
}


cumulative_regret <- cumsum(regrets)

cat(sprintf("Final cumulative regret: %.2f\n", cumulative_regret[T]))
cat("\nArm selection frequencies:\n")
print(table(chosen_arms))
barplot(table(chosen_arms), col = "purple",
        main = "Histogram of Chosen Arms",
        xlab = "Arm", ylab = "Number of times selected")

accuracy <- mean(chosen_arms == optimal_arms)
cat(sprintf("\nAccuracy of optimal arm selection: %.2f%%\n", 100 * accuracy))

# Plot convergence of θ_hat to θ*
plot(errors1, type = "l", main = "Convergence of θ_hat to θ*",
     xlab = "Round", ylab = "||θ_hat - θ*||2")

# Plot cumulative regret
plot(cumulative_regret, type = "l", main = "Cumulative Regret",
     xlab = "Round", ylab = "Regret")

# Plot fraction of optimal selections
optimal_fraction <- cumsum(chosen_arms == optimal_arms) / (1:T)
plot(optimal_fraction, type = "l", col = "blue",
     main = "Fraction of Optimal Arm Selections",
     xlab = "Round", ylab = "Fraction")

