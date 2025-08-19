library(MASS)

# Parameters
T <- 10000     
K <- 5
d <- 20
d_context <- 4
sigma <- 0.5
sigma0 <- 0.01

# LMC hyperparameters
h <- 1e-6
beta <- 1.0
K_lmc <- 20

set.seed(123)

# True parameter vector
theta_star <- rnorm(d, mean = 0, sd = sigma0)

# Feature map
phi <- function(X, i) {
  if (length(i) == 0 || i < 1 || i > K) stop("Invalid arm index in phi()")
  vec <- rep(0, d)
  start_idx <- (i - 1) * d_context + 1
  end_idx <- start_idx + d_context - 1
  vec[start_idx:end_idx] <- X
  return(vec)
}

# Storage
regrets <- numeric(T)
param_error <- numeric(T)     # posterior convergence measure
pred_error <- numeric(T)      # prediction accuracy measure

# Data storage
X_hist <- list()
r_hist <- c()

# Initialize theta
theta_curr <- rep(0, d)

for (t in 1:T) {
  X_t <- rnorm(d_context, mean = 0, sd = 1)
  
  # Gradient of neg log posterior
  grad_loss <- function(theta) {
    grad <- theta / (sigma0^2)
    if (length(r_hist) > 0) {
      for (s in 1:length(r_hist)) {
        phi_s <- X_hist[[s]]
        err <- sum(phi_s * theta) - r_hist[s]
        grad <- grad + (1 / sigma^2) * phi_s * err
      }
    }
    return(grad)
  }
  
  # LMC updates
  for (k in 1:K_lmc) {
    noise <- rnorm(d, mean = 0, sd = 1)
    theta_curr <- theta_curr - h * grad_loss(theta_curr) + sqrt(2 * h / beta) * noise
  }
  
  # Action selection
  pred_rewards <- sapply(1:K, function(a) sum(phi(X_t, a) * theta_curr))
  a_t <- which.max(pred_rewards)
  
  # Reward
  eps_t <- rnorm(1, mean = 0, sd = sigma)
  r_t <- sum(phi(X_t, a_t) * theta_star) + eps_t
  
  # Optimal reward
  opt_rewards <- sapply(1:K, function(a) sum(phi(X_t, a) * theta_star))
  r_star_t <- max(opt_rewards)
  
  regrets[t] <- r_star_t - sum(phi(X_t, a_t) * theta_star)
  
  # Store data
  X_hist[[length(X_hist) + 1]] <- phi(X_t, a_t)
  r_hist <- c(r_hist, r_t)
  
  # --- New diagnostics ---
  
  # Posterior convergence (Euclidean distance between theta_curr and theta_star)
  param_error[t] <- sqrt(sum((theta_curr - theta_star)^2))
  
  # Prediction accuracy: average squared error on past history
  if (length(r_hist) > 0) {
    pred_error[t] <- mean(sapply(1:length(r_hist), function(s) {
      (sum(X_hist[[s]] * theta_curr) - sum(X_hist[[s]] * theta_star))^2
    }))
  } else {
    pred_error[t] <- NA
  }
}

cumulative_regret <- cumsum(regrets)

cat(sprintf("Final cumulative regret: %.2f\n", cumulative_regret[T]))

# --- Plots ---
plot(cumulative_regret, type="l", col="blue", 
     main="Cumulative Regret", xlab="Round", ylab="Regret")

plot(param_error, type="l", col="red", 
     main="Posterior Convergence of Parameters", 
     xlab="Round", ylab="||theta_curr - theta_star||2")

plot(pred_error, type="l", col="darkgreen", 
     main="Prediction Accuracy (MSE)", 
     xlab="Round", ylab="Prediction Error")

table_chosen <- table(sapply(1:T, function(t) which.max(sapply(1:K, function(a) sum(phi(rnorm(d_context), a) * theta_curr)))))

