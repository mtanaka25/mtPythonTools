import numpy as np
from copy import deepcopy
from scipy.stats import norm

class AR1_process:
    def __init__(self,
                rho = 0.9000, # AR coefficient
                sig = 0.0080, # size of shock
                varname = 'x',  # variable name
                is_stationary = True, # If False, approximate nonstationary distribution
                sig_init_val = 0.01 # s.d. of initial value
                ):
        self.rho = rho
        self.sig = sig
        self.varname = varname
        self.is_stationary = is_stationary
        if not is_stationary:
            self.sig_x0 = sig_init_val
        
    
    def discretize(self,
                   method,
                   N = 100, # number of grid points
                   Omega = 3, # scale parameter of grid range (needed for Tauchen method)
                   approx_horizon = 10, # # of periods to be approximated (needed for non-stationary case)
                   is_write_out_result = True,
                   is_quiet = False):
        if self.is_stationary:
            if method in ['tauchen', 'Tauchen', 'T', 't']:
                if not is_quiet:
                    print("Discretizing the AR(1) process by Tauchen method...\n")
                self._tauchen_discretize(N, Omega, is_write_out_result)
            elif method in ['rouwenhorst', 'Rouwenhorst', 'R', 'r']:
                if not is_quiet:
                    print("Discretizing the AR(1) process by Rouwenhorst method...\n")
                self._rouwenhorst_discretize(N, is_write_out_result)
            else:
                raise Exception('"method" must be "Tauchen" or "Rouwenhorst."')
        else:
            if method in ['tauchen', 'Tauchen', 'T', 't']:
                if not is_quiet:
                    print("Discretizing the non-stationary AR(1) process by Tauchen method...\n")
                self._tauchen_discretize_nonstationary(N, Omega, approx_horizon, is_write_out_result)
            elif method in ['rouwenhorst', 'Rouwenhorst', 'R', 'r']:
                if not is_quiet:
                    print("Discretizing the non-stationary A(1) process by Rouwenhorst method...\n")
                self._rouwenhorst_discretize_nonstationary(N, approx_horizon, is_write_out_result)
            else:
                raise Exception('"method" must be "Tauchen" or "Rouwenhorst."')
    
    def _tauchen_discretize(self, N, Omega, is_write_out_result):
        # nested function to compute i-j element of the transition matrix
        def tauchen_trans_mat_ij(i, j, x_grid, h):
            if j == 0:
                trans_mat_ij = norm.cdf((x_grid[j] - self.rho*x_grid[i] + h/2)/self.sig)
            elif j == (N-1):
                trans_mat_ij = 1 - norm.cdf((x_grid[j] - self.rho*x_grid[i] - h/2)/self.sig)
            else:
                trans_mat_ij = ( norm.cdf((x_grid[j] - self.rho*x_grid[i] + h/2)/self.sig)
                               - norm.cdf((x_grid[j] - self.rho*x_grid[i] - h/2)/self.sig))
            return trans_mat_ij
        
        # Prepare gird points
        sig_x  = self.sig * (1 - self.rho**2)**(-1/2)
        x_max  = Omega * sig_x
        x_grid = np.linspace(-x_max, x_max, N)
        
        # Calculate the step size
        h = (2 * x_max)/(N - 1)
        
        # Construct the transition matrix
        trans_mat = [
            [tauchen_trans_mat_ij(i, j, x_grid, h) for j in range(N)]
            for i in range(N)
            ]
            
        if is_write_out_result:
            np.savetxt('Tauchen_{0}_grid.csv'.format(self.varname), x_grid, delimiter=' & ',
                       fmt='%2.3f', newline=' \\\\\n')
            np.savetxt('Tauchen_trans_mat.csv', trans_mat, delimiter=' & ',
                       fmt='%2.3f', newline=' \\\\\n')
        
        # Store the result as the instance's attributes
        self.__dict__['{0}_grid'.format(self.varname)] = x_grid
        self.trans_mat, self.step_size = np.array(trans_mat), h
    
    def _tauchen_discretize_nonstationary(self, N, Omega, approx_horizon, is_write_out_result):
        # TODO! Add the script for non-stationary tauchen
        pass
    
    def _rouwenhorst_discretize(self, N, is_write_out_result):
        # Prepare gird points
        sig_x  = self.sig * (1 - self.rho**2)**(-1/2)
        x_max  = sig_x * np.sqrt(N - 1)
        x_grid = np.linspace(-x_max, x_max, N)
        
        # Calculate the step size
        h = (2 * x_max)/(N-1)
        
        # transition probability satisfying the moment condition on the conditional expectation
        pi = 0.5 * (1 + self.rho)
        
        # N = 2
        Pi_N = np.array([[pi, 1 - pi],
                         [1 - pi, pi]])
        
        for n in range(3, N+1, 1):
            Pi_pre = deepcopy(Pi_N)
            Pi_N1, Pi_N2, Pi_N3, Pi_N4 = \
                np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))
            
            Pi_N1[:n-1, :n-1] = Pi_N2[:n-1, 1:n] = \
                Pi_N3[1:n, 1:n] = Pi_N4[1:n, :n-1] = Pi_pre
            
            Pi_N = (pi * Pi_N1
                    + (1 - pi) * Pi_N2
                    + pi * Pi_N3
                    + (1 - pi) * Pi_N4
            )
            # Divide all but the top and bottom rows by two so that the
            # elements in each row sum to one (Kopecky & Suen[2010, RED]).
            Pi_N[1:-1, :] *= 0.5
            
        if is_write_out_result:
            np.savetxt('Rouwenhorst_{0}_grid.csv'.format(self.varname), x_grid, delimiter=' & ',
                       fmt='%2.3f', newline=' \\\\\n')
            np.savetxt('Rouwenhorst_trans_mat.csv', Pi_N, delimiter=' & ',
                       fmt='%2.3f', newline=' \\\\\n')
        
        # Store the result as the instance's attributes
        self.__dict__['{0}_grid'.format(self.varname)] = x_grid
        self.trans_mat, self.step_size = Pi_N, h
    
    def _rouwenhorst_discretize_nonstationary(self, N, approx_horizon, is_write_out_result):
        if is_write_out_result:
            print('Warning: Approximation for non-stationary AR(1) does not implement export of results.\n')
        # Calculate the time-varying s.d. of x
        sig_x_vec = np.zeros((approx_horizon,))
        sig_x_vec[0] = self.sig_x0
        for t in range(approx_horizon-1):
            sig_x_vec[t+1] = self.rho**2 * sig_x_vec[t]**2 + self.sig**2
        
        # Prepare list of gird points
        x_max_vec  = sig_x_vec * np.sqrt(N - 1)
        x_grid_list = [
            np.linspace(-x_max_vec[t], x_max_vec[t], N) for t in range(approx_horizon)
        ]
        
        # Calculate the step sizes
        h = (2 * x_max_vec)/(N-1)
        
        trans_mat_list = []
        # approximate transition matirix
        for t in range(approx_horizon - 1):
            # transition probability satisfying the moment condition on the conditional expectation
            pi_t = 0.5 * (1 + self.rho * sig_x_vec(t)/sig_x_vec(t+1))
            
            # starting from N = 2
            Pi_N = np.array([[    pi_t, 1 - pi_t],
                             [1 - pi_t,     pi_t]])
            
            # Recursively expand N
            for n in range(3, N+1, 1):
                Pi_pre = deepcopy(Pi_N)
                Pi_N1, Pi_N2, Pi_N3, Pi_N4 = \
                    np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))
            
                Pi_N1[:n-1, :n-1] = Pi_N2[:n-1, 1:n] = \
                    Pi_N3[1:n, 1:n] = Pi_N4[1:n, :n-1] = Pi_pre
                
                Pi_N = (pi_t * Pi_N1
                        + (1 - pi_t) * Pi_N2
                        + pi_t * Pi_N3
                        + (1 - pi_t) * Pi_N4
                        )
                
                # Divide all but the top and bottom rows by two so that the
                # elements in each row sum to one (Kopecky & Suen[2010, RED]).
                Pi_N[1:-1, :] *= 0.5
            trans_mat_list.append(Pi_N)
        
        # Store the result as the instance's attributes
        self.__dict__['{0}_grid_list'.format(self.varname)] = x_grid_list
        self.trans_mat_list, self.step_size = trans_mat_list, h
