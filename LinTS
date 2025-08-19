library(MASS)

# Parameters
T <- 10000
K <- 5
d <- 20
d_context <- 4
sigma <- 0.5
sigma0 <- 0.01

set.seed(123)  

# True parameter vector
theta_star <- rnorm(d, mean = 0, sd = sigma0)

# Feature map phi(X, i)
phi <- function(X, i) {
  vec <- rep(0, d)
  start_idx <- (i - 1) * d_context + 1
  end_idx <- start_idx + d_context - 1
  vec[start_idx:end_idx] <- X
  return(vec)
}

# Prior mean and variance
mu <- rep(0, d)
Sigma <- diag(sigma0^2, d)

# Storage
regrets <- numeric(T)

for (t in 1:T) {
  X_t <- rnorm(d_context, mean = 0, sd = 1)
  
  theta_sample <- MASS::mvrnorm(1, mu, Sigma)
  
  pred_rewards <- sapply(1:K, function(a) sum(phi(X_t, a) * theta_sample))
  a_t <- which.max(pred_rewards)
  
  eps_t <- rnorm(1, mean = 0, sd = sigma)
  r_t <- sum(phi(X_t, a_t) * theta_star) + eps_t
  
  # Optimal reward 
  opt_rewards <- sapply(1:K, function(a) sum(phi(X_t, a) * theta_star))
  r_star_t <- max(opt_rewards)
  
  regrets[t] <- r_star_t - sum(phi(X_t, a_t) * theta_star)
  
  # Posterior update
  phi_vec <- phi(X_t, a_t)
  Sigma_inv <- solve(Sigma)
  Sigma <- solve(Sigma_inv + (1 / sigma^2) * (phi_vec %*% t(phi_vec)))
  mu <- Sigma %*% (Sigma_inv %*% mu + (1 / sigma^2) * phi_vec * r_t)
}

cumulative_regret <- cumsum(regrets)

cat(sprintf("Final cumulative regret: %.2f\n", cumulative_regret[T]))
