import numpy as np
from copy import deepcopy
from scipy.stats import norm

class AR1_process:
    def __init__(self,
                rho = 0.9000, # AR coefficient
                sig = 0.0080, # size of shock
                varname = 'x'  # variable name
                ):
        self.rho = rho
        self.sig = sig
        self.varname = varname
    
    def discretize(self, 
                   method,
                   N = 100, # number of grid points
                   Omega = 3, # scale parameter for Tauchen's grid range
                   is_write_out_result = True,
                   is_quiet = False): 
        if method in ['tauchen', 'Tauchen', 'T', 't']:
            if not is_quiet:
                print("Discretizing the AR(1) process by Tauchen method...\n")
            self._tauchen_discretize(N, Omega, is_write_out_result)
        elif method in ['rouwenhorst', 'Rouwenhorst', 'R', 'r']:
            if not is_quiet:
                print("Discretizing the income process by Rouwenhorst method...\n")
            self._rouwenhorst_discretize(N, is_write_out_result)
        else:
            raise Exception('"method" must be "Tauchen" or "Rouwenhorst."')
    
    def _tauchen_discretize(self,N, Omega, is_write_out_result):
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
    
    def _rouwenhorst_discretize(self, N, is_write_out_result):
        # Prepare gird points
        sig_x  = self.sig * (1 - self.rho**2)**(-1/2)
        x_max  = sig_x * np.sqrt(N - 1)
        x_grid = np.linspace(-x_max, x_max, N)
        
        # Calculate the step size
        h = (2 * x_max)/(N-1)
        
        # parameter necessary for Rouwenhorst recursion
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
        self.__dict__['{0}_gird'.format(self.varname)] = x_grid
        self.trans_mat, self.step_size = Pi_N, h