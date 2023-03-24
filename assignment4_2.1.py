def mini_batch_gradient_descent(g, w, x_train, y_train, alpha, max_its, batch_size):
    """
    Performs mini-batch gradient descent on a dataset (x_train, y_train) using the gradient function g,
    with learning rate alpha, for a maximum number of iterations max_its, and a given batch size.
    """
    n_samples = x_train.shape[0]
    num_batches = int(np.ceil(n_samples / batch_size))
    
    # Check if batch size is a divisor of the number of samples
    if n_samples % batch_size != 0:
        print(f"Warning: batch size {batch_size} is not a divisor of {n_samples}.")
    
    # Flatten gradient function g
    g_flat = lambda w, x, y: np.ravel(g(w, x, y))
    
    # Initialize cost history
    cost_history = []
    
    for epoch in range(max_its):
        epoch_cost = 0
        for batch in range(num_batches):
            # Get current batch of samples
            start_idx = batch * batch_size
            end_idx = min((batch+1) * batch_size, n_samples)
            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Compute gradient and update weights
            grad = g_flat(w, x_batch, y_batch)
            w -= alpha * grad
            
            # Compute cost for current batch
            epoch_cost += np.sum((y_batch - x_batch @ w)**2) / n_samples
        
        # Append cost for epoch to history
        cost_history.append(epoch_cost)
        
        print(f"Epoch {epoch+1}/{max_its}: cost={epoch_cost:.4f}")
    
    return w, cost_history
