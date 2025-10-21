# Parameters
T <- 10000            # number of rounds
K <- 5                # number of arms
d_context <- 4        # context dimension
d <- K * d_context    # full parameter dimension

sigma0 <- 0.5        # prior sd
sigma  <- 1         # observation noise sd

# LMC hyperparameters
h     <- 1e-4
beta  <- 1.0
K_lmc <- 20

# SFG hyperparameters
eta        <- 1/(sigma^2)
lambda_sfg <- 0.5
b_cap      <- 1000
s_param    <- 5.0

set.seed(123)
phi_map <- function(X, a) 
{
  vec <- rep(0, d)
  st <- (a - 1) * d_context + 1
  end_idx <- st + d_context - 1
  vec[st:end_idx] <- X
  return(vec)
}

sigmoid_stable <- function(z) 
{
  ifelse(z >= 0, 1 / (1 + exp(-z)), exp(z) / (1 + exp(z)))
}

# Gradient 
grad_loss_sfg <- function(theta, A, b, ctx)
{
  grad_prior <- theta / (sigma0^2)
  grad_sqerr <- if (sum(A) > 0) eta * (A %*% theta - b) else rep(0, d)
  # SFG term (only depends on current context)
  vals <- sapply(1:K, function(a) sum(phi_map(ctx, a) * theta))
  a_star <- which.max(vals)
  f_star <- vals[a_star]
  phi_max <- phi_map(ctx, a_star)
  z <- s_param * (b_cap - f_star)
  grad_sfg <- - lambda_sfg * sigmoid_stable(z) * phi_max
  
  return(drop(grad_prior + grad_sqerr + grad_sfg))
}


# True parameter and storage
theta_star <- c(5,5,5,5)
theta_star <- c(theta_star, rnorm(d-4, mean = 2, sd = sigma0))
theta_curr <- rep(0, d)

regrets      <- numeric(T)
chosen_arms  <- integer(T)
optimal_arms <- integer(T)
errors1      <- numeric(T)

# Statistics for fast gradient updates
A <- matrix(0, d, d)  # ∑ φ φ^T
b <- rep(0, d)        # ∑ φ r


for (t in 1:T) 
{
  # New context
  X_t <- rnorm(d_context, mean = 0, sd = 1)
  
  # LMC updates
  for (k in 1:K_lmc) 
  {
    noise <- rnorm(d)
    grad <- grad_loss_sfg(theta_curr, A, b, X_t)
    theta_curr <- theta_curr - h * grad + sqrt(2*h/beta) * noise
  }
  
  # Predict rewards and choose arm
  pred_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_curr))
  a_t <- which.max(pred_rewards)
  chosen_arms[t] <- a_t  
  
  # True reward
  eps_t <- rnorm(1, mean = 0, sd = sigma)
  r_t <- sum(phi_map(X_t, a_t) * theta_star) + eps_t
  
  # Regret
  opt_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_star))
  opt_arm <- which.max(opt_rewards)
  optimal_arms[t] <- opt_arm
  regrets[t] <- max(opt_rewards) - sum(phi_map(X_t, a_t) * theta_star)
  
  # Update statistics
  phi_t <- phi_map(X_t, a_t)
  A <- A + tcrossprod(phi_t)  # φ φ^T
  b <- b + phi_t * r_t
  
  errors1[t] <- sqrt(sum((theta_curr - theta_star)^2))
}

cumulative_regret <- cumsum(regrets)

cat(sprintf("Final cumulative regret: %.2f\n", cumulative_regret[T]))

# Arm selection histogram
cat("\nArm selection frequencies:\n")
print(table(chosen_arms))
barplot(table(chosen_arms), col="purple",
        main="Histogram of Chosen Arms",
        xlab="Arm", ylab="Number of times selected")

# Accuracy of arm selection
accuracy <- mean(chosen_arms == optimal_arms)
cat(sprintf("\nAccuracy of optimal arm selection: %.2f%%\n", 100 * accuracy))

# convergence of θ_hat to θ*
plot(errors1, type = "l", main = "Convergence of θ_hat to θ*",
     xlab = "Round", ylab = "||θ_hat - θ*||2")

# cumulative regret
plot(cumulative_regret, type = "l", main = "Cumulative Regret",
     xlab = "Round", ylab = "Regret")

# fraction of times optimal arm was chosen over time
optimal_fraction <- cumsum(chosen_arms == optimal_arms) / (1:T)
plot(optimal_fraction, type = "l", col = "blue",
     main = "Fraction of Optimal Arm Selections", ylim = c(0,1),
     xlab = "Round", ylab = "Fraction")

