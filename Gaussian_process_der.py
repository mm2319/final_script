# Gaussian Process
import numpy as np
import numpy
class Kernel:
  def __init__(self):
    pass

  def __call__(self, x, y, theta: list=None):
    raise NotImplementedError()
class RBF(Kernel):
  def __call__(self, x, y, theta: list=None):
    """
    Given an input array x of size n and another input array y of 
    size m, compute the n by m kernel matrix whose (i, j)-th entry 
    is k(x_i, y_j).

    Args:
      x: (n,)
      y: (m,)
    
    Returns:
      k_mat: (n, m)
    """
    # unpack hyper-params
    self.theta1, self.theta2 = theta

    ### TODO ###
    k_mat = (self.theta1**2)*np.exp((-np.square(np.array([x]*(y.shape[0])).T - np.array([y]*(x.shape[0])))/(2*self.theta2)))
    ### END OF TODO ###

    return k_mat

rbf = RBF()

class RBF_full_diff(Kernel):
  def __call__(self, x, y, theta: list=None):
    """
    Given an input array x of size n and another input array y of 
    size m, compute the n by m kernel matrix whose (i, j)-th entry 
    is k(x_i, y_j).

    Args:
      x: (n,)
      y: (m,)
    
    Returns:
      k_mat: (n, m)
    """
    # unpack hyper-params
    self.theta1, self.theta2 = theta

    ### TODO ###
    k_mat = (self.theta1**2)*np.exp((-np.square(np.array([x]*(y.shape[0])).T - np.array([y]*(x.shape[0])))/(2*self.theta2)))*(-np.square(np.array([x]*(y.shape[0])).T -(self.theta1**2)*np.array([y]*(x.shape[0])))/(self.theta2**2))+ (self.theta1**2)*np.exp((-np.square(np.array([x]*(y.shape[0])).T - np.array([y]*(x.shape[0])))/(2*self.theta2)))
    ### END OF TODO ###

    return k_mat

class RBF_partial_diff(Kernel):
  def __call__(self, x, y, theta: list=None):
    """
    Given an input array x of size n and another input array y of 
    size m, compute the n by m kernel matrix whose (i, j)-th entry 
    is k(x_i, y_j).

    Args:
      x: (n,)
      y: (m,)
    
    Returns:
      k_mat: (n, m)
    """
    # unpack hyper-params
    self.theta1, self.theta2 = theta

    ### TODO ###
    k_mat = (self.theta1**2)*np.exp((-np.square(np.array([x]*(y.shape[0])).T - np.array([y]*(x.shape[0])))/(2*self.theta2)))*(-(np.array([x]*(y.shape[0])).T - np.array([y]*(x.shape[0])))/(self.theta2))
    ### END OF TODO ###

    return k_mat
rbf_fd = RBF_full_diff()
rbf_pd = RBF_partial_diff()
class GP_derivative:
  def __init__(self, kernel: callable, kernel_diff: callable):
    """
    Args:
      kernel:
      prior_mean:
    """
    self.k = kernel
    self.k_d = kernel_diff

  def predict(
      self, 
      x_star, 
      X: np.array=None, 
      y: np.array=None, 
      size: int=1,
      theta: list=None,
      sigma: float=0.,
    ):
    """
    Given observations (X, y) and test points x_star, fit a GP model
    and draw posterior samples for f(x_star) from the fitted model.

    Args:
      x_star: (n*,) array of feature values at which predictions
        for f(x_star) will be made.
      X: (n,) observed features.
      y: (n,) observed response variables.
      size: number of posterior samples drawn.
      theta: (n_hyperparams,) array of kernel hyper-parameters.

    Returns:
      y_star: (size, n*) array of posterior samples for f(y_star).
    """
    ### TODO ###
    # 1. compute 
    # - k(x*, X)
    # - k(X, x*)
    # - k(x*, x*)
    # - k(X, X) + sigma^2 I_n
    k_xs_x = rbf_pd(x_star, X, theta)
    k_x_xs = rbf_pd(X, x_star, theta)
    k_xs_xs = rbf_pd(x_star, x_star, theta)
    k_x_x = rbf(X, X, theta) + (sigma**2)*np.identity(rbf(X, X, theta).shape[1])
    cov_x_x = k_xs_xs - k_xs_x@np.linalg.inv(k_x_x)@rbf(X,x_star,theta)-rbf(x_star,X,theta)@np.linalg.inv(k_x_x)@k_x_xs

    ### TODO ###
    # 2. compute posterior means and covariance matrix
    posterior_mean = k_xs_x@np.linalg.inv(k_x_x)@y
    posterior_var = cov_x_x

    ### END OF TODO ###

    self.posterior_mean = posterior_mean
    self.posterior_var = posterior_var
  

    ### TODO ###
    # 3. draw posterior samples by using the posterior
    # mean and covariance matrix
    y_star = np.random.multivariate_normal(posterior_mean, posterior_var, size)
    
    ### END OF TODO ###

    return y_star
class GP:
  def __init__(self, kernel: callable, kernel_diff: callable):
    """
    Args:
      kernel:
      prior_mean:
    """
    self.k = kernel
    self.k_d = kernel_diff

  def predict(
      self, 
      x_star, 
      X: np.array=None, 
      y: np.array=None, 
      size: int=1,
      theta: list=None,
      sigma: float=0.,
    ):
    """
    Given observations (X, y) and test points x_star, fit a GP model
    and draw posterior samples for f(x_star) from the fitted model.

    Args:
      x_star: (n*,) array of feature values at which predictions
        for f(x_star) will be made.
      X: (n,) observed features.
      y: (n,) observed response variables.
      size: number of posterior samples drawn.
      theta: (n_hyperparams,) array of kernel hyper-parameters.

    Returns:
      y_star: (size, n*) array of posterior samples for f(y_star).
    """
 
    k_xs_x = rbf(x_star, X, theta)
    k_x_xs = rbf(X, x_star, theta)
    k_xs_xs = rbf(x_star, x_star, theta)
    k_x_x = rbf(X, X, theta) + (sigma**2)*np.identity(rbf(X, X, theta).shape[1])
    cov_x_x = k_xs_xs - k_xs_x@np.linalg.inv(k_x_x)@k_x_xs

    posterior_mean = k_xs_x@np.linalg.inv(k_x_x)@y
    posterior_var = cov_x_x

    self.posterior_mean = posterior_mean
    self.posterior_var = posterior_var

    y_star = np.random.multivariate_normal(posterior_mean, posterior_var, size)

    return y_star
  
  def loglikelihood(
      self, 
      x_star, 
      X: np.array=None, 
      y: np.array=None, 
      size: int=1,
      theta: list=None,
      sigma: float=0.,
    ):
    n = rbf(X, X, theta).shape[1]
    K = rbf(X, X, theta)+1.e-8*np.eye(n)
    alpha = np.linalg.inv(rbf(X, X, theta) + (sigma**2)*np.identity(rbf(X, X, theta).shape[1]))@y
    L= np.linalg.cholesky(K)
    return -0.5*y.T.dot(alpha) - np.sum(np.log(np.diag(L))) - 0.5*n*np.log(2*np.pi)
  