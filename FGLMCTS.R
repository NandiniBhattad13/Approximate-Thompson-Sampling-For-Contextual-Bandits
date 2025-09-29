# Parameters
T <- 10000          
K <- 5              
d_context <- 4      
d <- K * d_context  

sigma0 <- 0.01   # prior std
sigma <- 0.5     # noise std

# FG / LMC hyperparameters
h <- 1e-6
eta <- 1.0
beta <- 1.0
K_lmc <- 20      
lambda_fg <- 0.5
b_cap <- 1000

set.seed(123)

# true parameter vector
theta_star <- rnorm(d, mean = 0, sd = sigma0)

# Feature map
phi_map <- function(X, a) 
{
  vec <- rep(0, d)
  st <- (a - 1) * d_context + 1
  end_idx <- st + d_context - 1
  vec[st:end_idx] <- X
  return(vec)
}

# Storage
regrets <- numeric(T)
chosen_arms <- integer(T)
optimal_arms <- integer(T)
errors1 <- numeric(T)

theta_curr <- rep(0, d)   

# Accumulators
A <- matrix(0, d, d)   # sum φφᵀ
b_vec <- rep(0, d)     # sum φ r
Phi_list <- list()     # keep past φ's (for FG term only)

# Gradient function (efficient)
grad_loss_fg <- function(theta) 
{
  grad_prior <- theta / (sigma0^2)
  grad_sqerr <- 2 * eta * (A %*% theta - b_vec)
  
  grad_fg <- rep(0, d)
  if (length(Phi_list) > 0) 
  {
    # stack phi's once (matrix form)
    Phi_mat <- do.call(rbind, Phi_list)
    preds <- Phi_mat %*% theta
    mask <- as.numeric(preds < b_cap)
    grad_fg <- lambda_fg * colSums(Phi_mat * mask)
  }
  
  return(drop(grad_prior + grad_sqerr - grad_fg))
}

for (t in 1:T) 
{
  # Context
  X_t <- rnorm(d_context, mean = 0, sd = 1)
  
  # LMC updates
  for (k in 1:K_lmc) 
  {
    noise <- rnorm(d)
    theta_curr <- theta_curr - h * grad_loss_fg(theta_curr) + sqrt(2 * h / beta) * noise
  }
  
  # Action selection
  pred_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_curr))
  a_t <- which.max(pred_rewards)
  chosen_arms[t] <- a_t
  
  # Reward
  eps_t <- rnorm(1, mean = 0, sd = sigma)
  phi_t <- phi_map(X_t, a_t)
  r_t <- sum(phi_t * theta_star) + eps_t
  
  # Optimal reward
  opt_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_star))
  opt_arm <- which.max(opt_rewards)
  optimal_arms[t] <- opt_arm
  r_star_t <- max(opt_rewards)
  
  # Regret
  regrets[t] <- r_star_t - sum(phi_map(X_t, a_t) * theta_star)
  
  # Update accumulators
  A <- A + tcrossprod(phi_t)
  b_vec <- b_vec + phi_t * r_t
  Phi_list[[length(Phi_list) + 1]] <- phi_t  # keep φ for FG term
  
  # Error
  errors1[t] <- sqrt(sum((theta_curr - theta_star)^2))
}

# Evaluation
cumulative_regret <- cumsum(regrets)
cat(sprintf("Final cumulative regret: %.2f\n", cumulative_regret[T]))

cat("\nArm selection frequencies:\n")
print(table(chosen_arms))
barplot(table(chosen_arms), col="purple",
        main="Histogram of Chosen Arms",
        xlab="Arm", ylab="Number of times selected")

accuracy <- mean(chosen_arms == optimal_arms)
cat(sprintf("\nAccuracy of optimal arm selection: %.2f%%\n", 100 * accuracy))

plot(errors1, type = "l", main = "Convergence of θ_hat to θ*",
     xlab = "Round", ylab = "||θ_hat - θ*||2")

plot(cumulative_regret, type = "l", main = "Cumulative Regret",
     xlab = "Round", ylab = "Regret")

optimal_fraction <- cumsum(chosen_arms == optimal_arms) / (1:T)
plot(optimal_fraction, type = "l", col = "blue",
     main = "Fraction of Optimal Arm Selections",
     ylim = c(0,1),
     xlab = "Round", ylab = "Fraction")

