import numpy as np

class PiecewiseIntrpl: # piecewised linear interpolation on a 1D grid
    def __init__(self, x, fx):
        if x is list:
            x, fx = np.array(x), np.array(fx)
        if (len(x.shape) == 2) & (x.shape[0] < x.shape[1]):
            x, fx = x.T, fx.T
        
        self.x  = x
        self.fx = fx
    
    def __call__(self, x_new):
        if np.isscalar(x_new):
            fx_bar = self._fit_scalar(x_new)
        else:
            if x_new is list:
                x_new = np.array(x_new)
            
            is_transposed = False
            if (len(x_new.shape) == 2) & (x_new.shape[0] < x_new.shape[1]):
                x_new = x_new.T
                is_transposed = True
                
            fx_bar = self._fit_array(x_new)
            
            if is_transposed:
                fx_bar = fx_bar.T
        return fx_bar
    
    def _fit_scalar(self, x_new):
        j  = sum(self.x <= x_new)
        
        phi = np.zeros((len(self.x), ))
        if self.x[-1] <= x_new:
            if self.x[-1] == self.x[-2]
                phi[-1] = 1
            else:
                phi[-1]   = (x_new - self.x[-2])/(self.x[-1] - self.x[-2])
                phi[-2] = (self.x[-1] -  x_new)/(self.x[-1] - self.x[-2])
        elif x_new <= self.x[0]:
            if self.x[1] == self.x[0]
                phi[0] = 1
            else:
                phi[1]   = (x_new - self.x[0])/(self.x[1] - self.x[0])
                phi[0] = (self.x[1] -  x_new)/(self.x[1] - self.x[0])
        else:
            if self.x[j] == self.x[j-1]
                phi[j] = 1
            else:
                phi[j]   = (x_new - self.x[j-1])/(self.x[j] - self.x[j-1])
                phi[j-1] = (self.x[j] -  x_new)/(self.x[j] - self.x[j-1])
        
        fx_bar = np.sum(phi * self.fx)
        
        return fx_bar
        
    def _fit_array(self, x_new):
        fx_bar = [self._fit_scalar(x_new_i) for x_new_i in x_new]
        return fx_bar


class PiecewiseIntrpl_MeshGrid:
    def __init__(self, x1_grid, x2_grid, fx):
        self.x1_grid, self.x2_grid = x1_grid.flatten(), x2_grid.flatten()
        self.fx = fx
    
    def __call__(self, x1, x2):
        if np.isscalar(x1) & np.isscalar(x2):
            fx_bar = self._fit_single_point(x1, x2)
        else:
            fx_bar = self._fit_multiple_points(x1, x2)
        return fx_bar
    
    def _fit_single_point(self, x1_new, x2_new):
        def get_marginal_weight_vector(x, x_hat, j):
            phi = np.zeros((len(x), ))
            if x[-1] <= x_hat:
                if x[-1] == x[-2]
                    phi[-1] = 1
                else:
                    phi[-1] = (x_hat - x[-2])/(x[-1] - x[-2])
                    phi[-2] = (x[-1] -  x_hat)/(x[-1] - x[-2])
            elif x_hat <= x[0]:
                if x[1] == x[0]
                    phi[0] = 1
                else:
                    phi[1]   = (x_hat - x[0])/(x[1] - x[0])
                    phi[0] = (x[1] -  x_hat)/(x[1] - x[0])
            else:
                if x[j] == x[j-1]
                    phi[j] = 1
                else:
                    phi[j]   = (x_hat - x[j-1])/(x[j] - x[j-1])
                    phi[j-1] = (x[j] -  x_hat)/(x[j] - x[j-1])
            return phi       
        j1  = sum(self.x1_grid <= x1_new)
        j2  = sum(self.x2_grid <= x2_new)
        
        phi1 = get_marginal_weight_vector(self.x1_grid, x1_new, j1)
        phi2 = get_marginal_weight_vector(self.x2_grid, x2_new, j2)
        
        phi1 = phi1.reshape(-1, 1)
        phi2 = phi2.reshape(1, -1)
        phi  = phi1 @ phi2
        
        fx_bar = np.sum(phi * self.fx)
        
        return fx_bar
    
    def _fit_multiple_points(self, x1, x2):
        if np.isscalar(x1):
            x1 = [x1]
        if np.isscalar(x2):
            x2 = [x2]
             
        fx_bar = np.array([
            [self._fit_single_point(x1[i], x2[j]) for j in range(len(x2))]
            for i in range(len(x1))
            ])
        return fx_bar
    
    def calc_partial_derivative(self, x1, x2, dim=0, dx=1E-5):
        if dim == 0:
            def f_prime(x1_k, x2_l):
                f_plus  = self.__call__(x1_k+dx, x2_l)
                f_minus = self.__call__(x1_k-dx, x2_l)
                f_prm = (f_plus - f_minus)/(2*dx)
                return f_prm
        else:
            def f_prime(x1_k, x2_l):
                f_plus  = self.__call__(x1_k, x2_l+dx)
                f_minus = self.__call__(x1_k, x2_l-dx)
                f_prm = (f_plus - f_minus)/(2*dx)
                return f_prm
        if np.isscalar(x1):
            x1 = [x1]
        if np.isscalar(x2):
            x2 = [x2]
        f_prm_vec = np.array([
                [f_prime(x1[i], x2[j]) for j in range(len(x2))] for i in range(len(x1))
                ])
        return f_prm_vec


class RBFIntrpl: # radial basis function interpolation
    def __init__(self, x, fx, eps=0.01):
        self.eps = eps
        self.x   = x
        self._solve_for_omega(x, fx)
    
    def __call__(self, x_new):
        if np.isscalar(x_new):
            fx_bar = self._fit_single_point(x_new)
        else:
            fx_bar = self._fit_multiple_points(x_new)
        return fx_bar
    
    def _rbf(self, x_1, x_2=0): # radial basis function
        distance = abs(x_1 - x_2)
        phi = np.exp(- self.eps * distance**2)
        return phi
    
    def _solve_for_omega(self, x, fx):
        coef_mat = [
            [self._rbf(x[i], x[j]) for i in range(len(x))]
            for j in range(len(x))
            ]
        omega = np.linalg.solve(coef_mat, fx)
        self.omega = omega
    
    def _fit_single_point(self, x_new):
        fx_bar = [self._rbf(x_new, self.x[i])*self.omega[i]
                  for i in range(len(self.x))]
        fx_bar = np.sum(fx_bar)
        return fx_bar
    
    def _fit_multiple_points(self, x_new):
        fx_bar = [self._fit_single_point(x_new_i) for x_new_i in x_new]
        return fx_bar


class RBFIntrpl_MeshGrid: # radial basis function interpolation
    def __init__(self, x1_grid ,x2_grid, fx, eps=1):
        x1_grid = x1_grid.flatten()
        x2_grid = x2_grid.flatten()
        fx_flatten = sum((fx.T).tolist(), [])
        self.eps = eps
        self.x1_grid, self.x2_grid = x1_grid, x2_grid
        self._solve_for_omega(x1_grid, x2_grid, fx_flatten)      
    
    def __call__(self, x1, x2):
        if np.isscalar(x1) & np.isscalar(x2):
            fx_bar = self._fit_single_point(np.array([x1, x2]))
        else:
            fx_bar = self._fit_multiple_points(x1, x2)
        return fx_bar
    
    def _rbf(self, x, x0): # radial basis function
        distance = abs(x - x0)
        phi = np.exp(- self.eps * np.sum(distance**2))
        return phi
    
    def _solve_for_omega(self, x1, x2, fx):
        coef_mat = np.array([
            [self._rbf(np.array([x1[k], x2[l]]), np.array([x1[i], x2[j]])) for j in range(len(x2)) for i in range(len(x1))]
             for l in range(len(x2)) for k in range(len(x1))
            ])
        omega = np.linalg.solve(coef_mat, fx)
        coef_inv = np.linalg.inv(coef_mat)
        omega2 = coef_inv @ np.array(fx).reshape(-1, 1)
        self.coef_mat = coef_mat
        self.omega = omega
        self.omega2 = omega2
    
    def _fit_single_point(self, x_new):
        x_new = np.array(x_new)
        phi_vec = np.array([
                   self._rbf(x_new, np.array([self.x1_grid[i], self.x2_grid[j]]))
                   for j in range(len(self.x2_grid)) for i in range(len(self.x1_grid)) 
                   ])        
        fx_bar = np.sum(phi_vec * self.omega)
        return fx_bar
    
    def _fit_multiple_points(self, x1, x2):
        if np.isscalar(x1):
            x1 = [x1]
        if np.isscalar(x2):
            x2 = [x2]
        fx_bar = np.array([
            [self._fit_single_point([x1[i], x2[j]]) for j in range(len(x2))]
            for i in range(len(x1))
            ])
        return fx_bar
    
    def calc_partial_derivative(self, 
                                x1, # point(s) on which the partial derivative is calculated (dim 0)
                                x2, # point(s) on which the partial derivative is calculated (dim 1)
                                dim = 0 # For which variable, is partical derivative calculated?
                                ):
        if dim == 0:
            def f_prime(x1_k, x2_l):
                x_new = np.array([x1_k, x2_l])
                phi_prm_vec = np.array([
                           -2 * self.eps * (x1_k - self.x1_grid[i])*self._rbf(x_new, np.array([self.x1_grid[i], self.x2_grid[j]]))
                           for j in range(len(self.x2_grid)) for i in range(len(self.x1_grid)) 
                           ])
                f_prm = np.sum(phi_prm_vec * self.omega)
                return f_prm
        else:
            def f_prime(x1_k, x2_l):
                x_new = np.array([x1_k, x2_l])
                phi_prm_vec = np.array([
                           -2 * self.eps * (x2_l - self.x2_grid[j])*self._rbf(x_new, np.array([self.x1_grid[i], self.x2_grid[j]]))
                           for j in range(len(self.x2_grid)) for i in range(len(self.x1_grid)) 
                           ])
                f_prm = np.sum(phi_prm_vec * self.omega)
                return f_prm
        if np.isscalar(x1):
            x1 = [x1]
        if np.isscalar(x2):
            x2 = [x2]
        f_prm_vec = np.array([
                [f_prime(x1[i], x2[j]) for j in range(len(x2))] for i in range(len(x1))
                ])
        return f_prm_vec