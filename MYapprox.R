# Parameters
T <- 10000          
K <- 5              
d_context <- 4      
d <- K * d_context  

sigma0 <- 0.01
sigma  <- 0.5

# FG / LMC hyperparameters
h <- 1e-6
eta <- 1/(sigma^2)
beta <- 1.0
K_lmc <- 20      
lambda_fg <- 0.5
b_cap <- 0.5
lambda_my <- 5.0   # smoothing parameter for Moreau–Yosida approx

set.seed(123)
theta_star <- rnorm(d, mean = 0, sd = sigma0)

# feature map
phi_map <- function(X, a) 
{
  vec <- rep(0, d)
  st <- (a - 1) * d_context + 1
  end_idx <- st + d_context - 1
  vec[st:end_idx] <- X
  vec
}

# ----- Moreau–Yosida smoothed derivative -----
# derivative of [-min(x,a)] approx
grad_my_envelope <- function(x, a, lam) 
{
  ifelse(
    x <= a - lam, -1,
    ifelse(x >= a, 0, (x - a) / lam)
  )
}

# ----- Gradient of smoothed FGTS -----
grad_loss_myfg <- function(theta) 
{
  grad_prior <- theta / (sigma0^2)
  grad_sqerr <- 2 * eta * (A %*% theta - b_vec)
  
  grad_fg <- rep(0, d)
  if (length(Phi_list) > 0) 
  {
    Phi_mat <- do.call(rbind, Phi_list)
    preds <- Phi_mat %*% theta
    # smooth derivative term
    gvals <- grad_my_envelope(preds, b_cap, lambda_my)
    grad_fg <- lambda_fg * colSums(Phi_mat * as.numeric(gvals))
  }
  
  drop(grad_prior + grad_sqerr + grad_fg)
}

# storage
regrets <- numeric(T)
chosen_arms <- integer(T)
optimal_arms <- integer(T)
errors1 <- numeric(T)

theta_curr <- rep(0, d)
A <- matrix(0, d, d)
b_vec <- rep(0, d)
Phi_list <- list()

for (t in 1:T) 
{
  X_t <- rnorm(d_context)
  
  # LMC updates
  for (k in 1:K_lmc) 
  {
    noise <- rnorm(d)
    theta_curr <- theta_curr - h * grad_loss_myfg(theta_curr) + sqrt(2 * h / beta) * noise
  }
  
  # select arm
  pred_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_curr))
  a_t <- which.max(pred_rewards)
  chosen_arms[t] <- a_t
  
  # reward + regret
  eps_t <- rnorm(1, sd = sigma)
  phi_t <- phi_map(X_t, a_t)
  r_t <- sum(phi_t * theta_star) + eps_t
  
  opt_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_star))
  optimal_arms[t] <- which.max(opt_rewards)
  regrets[t] <- max(opt_rewards) - sum(phi_map(X_t, a_t) * theta_star)
  
  # update stats
  A <- A + tcrossprod(phi_t)
  b_vec <- b_vec + phi_t * r_t
  Phi_list[[length(Phi_list) + 1]] <- phi_t
  
  errors1[t] <- sqrt(sum((theta_curr - theta_star)^2))
}

# --- Evaluation ---
cumulative_regret <- cumsum(regrets)
cat(sprintf("Final cumulative regret: %.2f\n", cumulative_regret[T]))

cat("\nArm selection frequencies:\n")
print(table(chosen_arms))
barplot(table(chosen_arms), col="purple",
        main="Histogram of Chosen Arms",
        xlab="Arm", ylab="Number of times selected")

accuracy <- mean(chosen_arms == optimal_arms)
cat(sprintf("\nAccuracy of optimal arm selection: %.2f%%\n", 100 * accuracy))

plot(errors1, type="l", main="Convergence of θ̂ to θ*",
     xlab="Round", ylab="||θ̂−θ*||₂")

plot(cumulative_regret, type="l", main="Cumulative Regret",
     xlab="Round", ylab="Regret")

optimal_fraction <- cumsum(chosen_arms == optimal_arms) / (1:T)
plot(optimal_fraction, type="l", col="blue",
     main="Fraction of Optimal Arm Selections",
     ylim=c(0,1), xlab="Round", ylab="Fraction")
