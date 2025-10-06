lambda_fg <- 0.5
b_cap <- 1000
K <- 5
d_context <- 4
d <- K * d_context
sigma0 <- 0.01
n_points <- 2000 


set.seed(123)
theta <- rnorm(d, mean = 0, sd = sigma0)

phi_map <- function(X, a, d, d_context) 
{
  vec <- rep(0, d)
  st <- (a - 1) * d_context + 1
  end_idx <- st + d_context - 1
  vec[st:end_idx] <- X
  return(vec)
}

# 3. Generate a matrix of phi vectors ('x_matrix')
# Each row will be a correctly structured sparse vector
phi_matrix <- matrix(0, nrow = n_points, ncol = d)

# We scale the input context X_t so the final dot product is in a visible range
# The variance of the dot product is roughly d_context * var(X) * var(theta)
# We target a standard deviation of about b_cap / 3 to see the kink.
scaling_factor <- (b_cap / 3) / sqrt(d_context * 1 * sigma0^2)

for (i in 1:n_points) 
{
  # Generate a random context and scale it
  context_xt <- rnorm(d_context)*scaling_factor
  
  # Choose a random arm
  arm <- sample(1:K, 1)
  
  # Create the full 20-element sparse feature vector
  phi_matrix[i, ] <- phi_map(context_xt, arm, d, d_context)
}

# 4. Generate f_theta_x values by taking the dot product
f_theta_x <- phi_matrix %*% theta

# Sort the results for a clean, continuous line plot
f_theta_x_sorted <- sort(f_theta_x)

# 5. Calculate the corresponding y-values for the likelihood term
likelihood_term <- lambda_fg * pmin(b_cap, f_theta_x_sorted)

# 6. Create the plot
plot(f_theta_x_sorted, likelihood_term, 
     type = "l", 
     col = "blue", 
     lwd = 2,
     xlab = expression(f[theta](x) == phi(X,a)^T * theta),
     ylab = expression(lambda %.% minimum(b, f[theta](x))),
     panel.first = grid())

# Add a vertical line to show the non-differentiable point
abline(v = b_cap, col = "red", lty = 2, lwd = 2)

# Add a legend to the plot
legend("topleft", 
       legend = c(expression(lambda %.% minimum(b, f[theta](x))), "Non-differentiable point"),
       col = c("blue", "red"), 
       lty = c(1, 2), 
       lwd = c(2, 2))

