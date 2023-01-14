# random search function
import numpy as np
import matplotlib.pyplot as plt
def random_search(g,alpha_choice,max_its,w,num_samples):
    # run random search
    w_history = [w[0]] # container for w history
    cost_history = [] # container for corresponding cost function history
    alpha = 0

    for k in range(1,max_its+1):
        print(k)
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice

        directions = np.random.randn(num_samples, w.shape[0], w.shape[1])
        for i, direction in enumerate(directions):
            direction_norm = direction / np.linalg.norm(direction)
            directions[i] = direction_norm

        # evaluate cost function at current w
        w_candidates = w + alpha * directions
        vals = [g(w_c[0]) for w_c in w_candidates]
        # find best candidate
        cost_history.append(min(vals))

        best_idx = np.argmin(vals)
        w = w_candidates[best_idx]
        w_history.append(w[0])


    return w_history, cost_history

# def g(x):
#     x = np.linalg.norm(x)
#     return x

g = lambda w: np.dot(w.T,w) + 2
alpha_choice = 0.3; w = np.array([[3,4]]); num_samples = 1000; max_its = 5;

w_history, cost_history = random_search(g=g,alpha_choice=alpha_choice,max_its=max_its,w=w,num_samples=num_samples)

# print(w_history)
print(cost_history)
x_w = [pair[0] for pair in w_history]
y_w = [pair[1] for pair in w_history]

plt.scatter(x_w,y_w)
plt.show()

plt.plot(cost_history)
plt.show()

