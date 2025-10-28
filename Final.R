###############################################################
# Combined Simulation: FGLMCTS, SFGLMCTS, MYFGTS, FGTS_Barker
###############################################################

# Simulation setup
T <- 1000              # total rounds
K <- 5                 # number of arms
d_context <- 4         # context dimension
d <- K * d_context     # total parameter dimension

# Noise parameters
sigma0 <- 0.5          # prior std deviation
sigma  <- 1.0          # reward noise std deviation

# Random seed for reproducibility
set.seed(123)

# True parameter vector
theta_star <- c(rep(5, 4), rnorm(d - 4, mean = 2, sd = sigma0))

#############################
# Global Algorithm Hyperparameters
#############################

# Shared across all algorithms
h     <- 1e-4           # step size for Langevin updates
eta   <- 1 / (sigma^2)  # noise scaling for reward
beta  <- 1.0            # inverse temperature for Langevin
b_cap <- 0.5            # clipping threshold

# Algorithm-specific parameters
K_lmc <- 20             # number of Langevin steps (common)
lambda_fg <- 0.5        # regularization weight

# MYFGTS extra
lambda_my <- 5.0

# SFGLMCTS extra
s_param <- 5

# FGTS_Barker extra
h_barker <- 0.1
K_barker <- 1

#############################
# Shared Helper Functions
#############################

# Context-to-feature mapping
phi_map <- function(X, a) 
{
  vec <- rep(0, d)
  st <- (a - 1) * d_context + 1
  end_idx <- st + d_context - 1
  vec[st:end_idx] <- X
  vec
}

# Sigmoid (numerically stable)
sigmoid_stable <- function(z) 
{
  ifelse(z >= 0, 1 / (1 + exp(-z)), exp(z) / (1 + exp(z)))
}

# Smooth envelope gradient for MYFGTS
grad_my_envelope <- function(x, a, lam) 
{
  ifelse(x <= a - lam, -1,
         ifelse(x >= a, 0, (x - a) / lam))
}

###############################################################
# Helper plotting function
###############################################################
plot_all_metrics <- function(res_list) 
{
  par(mfrow = c(3, 2))
  for (name in names(res_list)) 
  {
    res <- res_list[[name]]
    plot(res$cum_regret, type = "l", main = paste("Cumulative Regret:", name),
         xlab = "Round", ylab = "Cumulative Regret")
    plot(res$opt_fraction, type = "l", col = "blue", ylim = c(0,1),
         main = paste("Optimal Arm Fraction:", name),
         xlab = "Round", ylab = "Fraction")
    plot(res$param_error, type = "l", col = "darkgreen",
         main = paste("Convergence of θ̂:", name),
         xlab = "Round", ylab = "||θ̂ - θ*||₂")
  }
}

###############################################################
# Algorithm 1: FGLMCTS
###############################################################
run_FGLMCTS <- function() 
{
  theta_curr <- rep(0, d)
  A <- matrix(0, d, d)
  b_vec <- rep(0, d)
  Phi_list <- list()
  
  grad_loss_fg <- function(theta) 
  {
    grad_prior <- theta / (sigma0^2)
    grad_sqerr <- eta * (A %*% theta - b_vec)
    grad_fg <- rep(0, d)
    if (length(Phi_list) > 0) 
    {
      Phi_mat <- do.call(rbind, Phi_list)
      preds <- Phi_mat %*% theta
      mask <- as.numeric(preds < b_cap)
      grad_fg <- lambda_fg * colSums(Phi_mat * mask)
    }
    drop(grad_prior + grad_sqerr - grad_fg)
  }
  
  regrets <- chosen_arms <- optimal_arms <- errors <- numeric(T)
  
  for (t in 1:T) {
    X_t <- rnorm(d_context)
    for (k in 1:K_lmc) {
      noise <- rnorm(d)
      theta_curr <- theta_curr - h * grad_loss_fg(theta_curr) +
        sqrt(2 * h / beta) * noise
    }
    pred_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_curr))
    a_t <- which.max(pred_rewards)
    eps_t <- rnorm(1, 0, sigma)
    phi_t <- phi_map(X_t, a_t)
    r_t <- sum(phi_t * theta_star) + eps_t
    opt_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_star))
    opt_arm <- which.max(opt_rewards)
    
    regrets[t] <- max(opt_rewards) - sum(phi_t * theta_star)
    chosen_arms[t] <- a_t
    optimal_arms[t] <- opt_arm
    errors[t] <- sqrt(sum((theta_curr - theta_star)^2))
    
    A <- A + tcrossprod(phi_t)
    b_vec <- b_vec + phi_t * r_t
    Phi_list[[length(Phi_list)+1]] <- phi_t
  }
  
  list(
    cum_regret = cumsum(regrets),
    opt_fraction = cumsum(chosen_arms == optimal_arms) / (1:T),
    param_error = errors
  )
}

###############################################################
# Algorithm 2: SFGLMCTS
###############################################################
run_SFGLMCTS <- function() 
{
  theta_curr <- rep(0, d)
  A <- matrix(0, d, d)
  b <- rep(0, d)
  
  grad_loss_sfg <- function(theta, A, b, ctx) 
  {
    grad_prior <- theta / (sigma0^2)
    grad_sqerr <- if (sum(A) > 0) eta * (A %*% theta - b) else rep(0, d)
    vals <- sapply(1:K, function(a) sum(phi_map(ctx, a) * theta))
    a_star <- which.max(vals)
    f_star <- vals[a_star]
    phi_max <- phi_map(ctx, a_star)
    z <- s_param * (b_cap - f_star)
    grad_sfg <- - lambda_fg * sigmoid_stable(z) * phi_max
    drop(grad_prior + grad_sqerr + grad_sfg)
  }
  
  regrets <- chosen_arms <- optimal_arms <- errors <- numeric(T)
  
  for (t in 1:T) 
  {
    X_t <- rnorm(d_context)
    for (k in 1:K_lmc) 
    {
      noise <- rnorm(d)
      grad <- grad_loss_sfg(theta_curr, A, b, X_t)
      theta_curr <- theta_curr - h * grad +
        sqrt(2 * h / beta) * noise
    }
    pred_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_curr))
    a_t <- which.max(pred_rewards)
    eps_t <- rnorm(1, 0, sigma)
    phi_t <- phi_map(X_t, a_t)
    r_t <- sum(phi_t * theta_star) + eps_t
    opt_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_star))
    opt_arm <- which.max(opt_rewards)
    
    regrets[t] <- max(opt_rewards) - sum(phi_t * theta_star)
    chosen_arms[t] <- a_t
    optimal_arms[t] <- opt_arm
    errors[t] <- sqrt(sum((theta_curr - theta_star)^2))
    
    A <- A + tcrossprod(phi_t)
    b <- b + phi_t * r_t
  }
  
  list(
    cum_regret = cumsum(regrets),
    opt_fraction = cumsum(chosen_arms == optimal_arms) / (1:T),
    param_error = errors
  )
}

###############################################################
# Algorithm 3: MYFGTS
###############################################################
run_MYFGTS <- function() {
  theta_curr <- rep(0, d)
  A <- matrix(0, d, d)
  b_vec <- rep(0, d)
  Phi_list <- list()
  
  grad_loss_myfg <- function(theta) {
    grad_prior <- theta / (sigma0^2)
    grad_sqerr <- eta * (A %*% theta - b_vec)
    grad_fg <- rep(0, d)
    if (length(Phi_list) > 0) {
      Phi_mat <- do.call(rbind, Phi_list)
      preds <- Phi_mat %*% theta
      gvals <- grad_my_envelope(preds, b_cap, lambda_my)
      grad_fg <- lambda_fg * colSums(Phi_mat * as.numeric(gvals))
    }
    drop(grad_prior + grad_sqerr + grad_fg)
  }
  
  regrets <- chosen_arms <- optimal_arms <- errors <- numeric(T)
  
  for (t in 1:T) {
    X_t <- rnorm(d_context)
    for (k in 1:K_lmc) {
      noise <- rnorm(d)
      theta_curr <- theta_curr - h * grad_loss_myfg(theta_curr) +
        sqrt(2 * h / beta) * noise
    }
    pred_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_curr))
    a_t <- which.max(pred_rewards)
    eps_t <- rnorm(1, 0, sigma)
    phi_t <- phi_map(X_t, a_t)
    r_t <- sum(phi_t * theta_star) + eps_t
    opt_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_star))
    opt_arm <- which.max(opt_rewards)
    
    regrets[t] <- max(opt_rewards) - sum(phi_t * theta_star)
    chosen_arms[t] <- a_t
    optimal_arms[t] <- opt_arm
    errors[t] <- sqrt(sum((theta_curr - theta_star)^2))
    
    A <- A + tcrossprod(phi_t)
    b_vec <- b_vec + phi_t * r_t
    Phi_list[[length(Phi_list)+1]] <- phi_t
  }
  
  list(
    cum_regret = cumsum(regrets),
    opt_fraction = cumsum(chosen_arms == optimal_arms) / (1:T),
    param_error = errors
  )
}

###############################################################
# Algorithm 4: MYFGTS_Barker
###############################################################
run_MYFGTS_Barker <- function() 
{
  theta_curr <- rep(0, d)
  A <- matrix(0, d, d)
  b_vec <- rep(0, d)
  Phi_list <- list()
  
  grad_loss_myfg <- function(theta) 
  {
    grad_prior <- theta / (sigma0^2)
    grad_sqerr <- eta * (A %*% theta - b_vec)
    
    grad_fg <- rep(0, d)
    if (length(Phi_list) > 0) 
    {
      Phi_mat <- do.call(rbind, Phi_list)
      preds <- Phi_mat %*% theta
      gvals <- grad_my_envelope(preds, b_cap, lambda_my)
      grad_fg <- lambda_fg * colSums(Phi_mat * as.numeric(gvals))
    }
    drop(grad_prior + grad_sqerr + grad_fg)
  }
  
  regrets <- chosen_arms <- optimal_arms <- errors <- numeric(T)
  
  for (t in 1:T) 
  {
    X_t <- rnorm(d_context)
    
    # Barker dynamics
    for (k in 1:K_barker) 
    {
      grad_curr <- grad_loss_myfg(theta_curr)
      w <- rnorm(d, 0, h_barker)
      p <- plogis(-0.5 * grad_curr * w)
      b <- ifelse(runif(d) < p, 1, -1)
      theta_curr <- theta_curr + b * w
    }
    
    pred_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_curr))
    a_t <- which.max(pred_rewards)
    chosen_arms[t] <- a_t
    
    eps_t <- rnorm(1, sd = sigma)
    phi_t <- phi_map(X_t, a_t)
    r_t <- sum(phi_t * theta_star) + eps_t
    
    opt_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_star))
    optimal_arms[t] <- which.max(opt_rewards)
    regrets[t] <- max(opt_rewards) - sum(phi_map(X_t, a_t) * theta_star)
    
    A <- A + tcrossprod(phi_t)
    b_vec <- b_vec + phi_t * r_t
    Phi_list[[length(Phi_list) + 1]] <- phi_t
    errors[t] <- sqrt(sum((theta_curr - theta_star)^2))
  }
  
  # --- Outputs ---
  list(
    cum_regret    = cumsum(regrets),
    opt_fraction  = cumsum(chosen_arms == optimal_arms) / (1:T),
    param_error   = errors
  )
}

###############################################################
# Run 200 simulations and average results
###############################################################
n_runs <- 200

# Helper function to safely accumulate results
accumulate_results <- function(res_list_acc, res_new) 
{
  if (is.null(res_list_acc)) return(res_new)
  for (nm in names(res_new)) 
  {
    res_list_acc[[nm]] <- res_list_acc[[nm]] + res_new[[nm]]
  }
  res_list_acc
}

# Initialize cumulative storage for averages
avg_results <- list(
  FGLMCTS = NULL,
  SFGLMCTS = NULL,
  MYFGTS = NULL,
  MYFGTS_Barker = NULL
)

for (i in 1:n_runs) 
{
  if (i %% 10 == 0) cat("Running simulation", i, "of", n_runs, "...\n")
  
  res_iter <- list(
    FGLMCTS       = run_FGLMCTS(),
    SFGLMCTS      = run_SFGLMCTS(),
    MYFGTS        = run_MYFGTS(),
    MYFGTS_Barker = run_MYFGTS_Barker()
  )
  
  for (nm in names(avg_results)) 
  {
    avg_results[[nm]] <- accumulate_results(avg_results[[nm]], res_iter[[nm]])
  }
}

# Average over n_runs
for (nm in names(avg_results)) 
{
  for (metric in names(avg_results[[nm]])) 
  {
    avg_results[[nm]][[metric]] <- avg_results[[nm]][[metric]] / n_runs
  }
}

###############################################################
# Visualization of averaged metrics
###############################################################
plot_all_metrics(avg_results)

# --- Combined Boxplots ---
regret_df <- data.frame(
  FGLMCTS      = avg_results$FGLMCTS$cum_regret,
  SFGLMCTS     = avg_results$SFGLMCTS$cum_regret,
  MYFGTS       = avg_results$MYFGTS$cum_regret,
  MYFGTS_Barker = avg_results$MYFGTS_Barker$cum_regret
)

par(mfrow = c(1, 1))
boxplot(regret_df,
        main = "Average Cumulative Regret (200 runs)",
        ylab = "Cumulative Regret",
        col = c("tomato", "skyblue", "lightgreen", "plum"))

optfrac_df <- data.frame(
  FGLMCTS      = avg_results$FGLMCTS$opt_fraction,
  SFGLMCTS     = avg_results$SFGLMCTS$opt_fraction,
  MYFGTS       = avg_results$MYFGTS$opt_fraction,
  MYFGTS_Barker = avg_results$MYFGTS_Barker$opt_fraction
)

boxplot(optfrac_df,
        main = "Average Optimal Arm Fraction (200 runs)",
        ylab = "Optimal Arm Fraction",
        ylim = c(0, 1),
        col = c("tomato", "skyblue", "lightgreen", "plum"))
