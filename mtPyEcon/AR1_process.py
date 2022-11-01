import numpy as np
from scipy.stats import norm

class AR1_process:
    """
    This class discretizes the AR1 process by Tauchen or Rouwenhorst methods.
    """
    def __init__(self,
                rho = 0.9000,
                sig = 0.0080,
                varname = 'x',
                is_stationary = True,
                sig_init_val = 0.01
                ):
        """
        This class discretizes the AR1 process by Tauchen or Rouwenhorst methods.
        ----------
        rho : float, optional
            AR1 coefficient, by default 0.9
        sig : float, optional
            standard deviations of the exogenous shock, by default 0.008
        varname: string, optional
            variable name, by default 'x'
        is_stationary: bool, optional
            If true, stationary discretization is applied. Otherwise, non-stationary discretization
            is applied. By default True.
        sig_init_val : float, optional
            standard deviations of the AR1 variable (that is, z0). While this is necessary
            for non-statonary discretization, this is not used in stationary discretization.
            by default 0.01
        """
        self.rho = rho
        self.sig = sig
        self.varname = varname
        self.is_stationary = is_stationary
        if not is_stationary:
            self.sig_x0 = sig_init_val
    
    def discretize(self,
                   method,
                   N = 100,
                   Omega = 3.,
                   approx_horizon = 10,
                   is_write_out_result = True,
                   is_quiet = False):
        """
        This method implements discretization.
        Parameters
        ----------
        method : string
            method of discretization.
            For Tauchen's method, input "tauchen", "Tauchen", "T", or "t"
            For Rouwenhorst's method, input "rouwenhorst", "ROuwenhorst", "R", or "r"
        N : int, optional
            # of grid points, by default 100
        Omega : float, optional
            range parameter of Tauchen's method, by default 3.0
        approx_horizon : int, optional
            # of periods to be descretized. This is used only in non-statonary discretization.
            by default 10
        is_write_out_result : bool, optional
            If True, obtained grid points and transition matrix will be written out in csv file,
            by default True
        is_quiet : bool, optional
            If True, messages will not be shown in Terminal, by default False
        """
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
                    print("Discretizing the non-stationary AR(1) process by Rouwenhorst method...\n")
                self._rouwenhorst_discretize_nonstationary(N, approx_horizon, is_write_out_result)
            else:
                raise Exception('"method" must be "Tauchen" or "Rouwenhorst."')
    
    def _tauchen_discretize(self, N, Omega, is_write_out_result):
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
        def tauchen_trans_mat_ij(i, j, x_grid_post, x_grid_pre, h):
            if j == 0:
                trans_mat_ij = norm.cdf((x_grid_post[j] - self.rho*x_grid_pre[i] + h/2)/self.sig)
            elif j == (N-1):
                trans_mat_ij = 1 - norm.cdf((x_grid_post[j] - self.rho*x_grid_pre[i] - h/2)/self.sig)
            else:
                trans_mat_ij = ( norm.cdf((x_grid_post[j] - self.rho*x_grid_pre[i] + h/2)/self.sig)
                                - norm.cdf((x_grid_post[j] - self.rho*x_grid_pre[i] - h/2)/self.sig))
            return trans_mat_ij
        if is_write_out_result:
            print('Warning: Approximation for non-stationary AR(1) does not implement export of results.\n')
        # Calculate the time-varying s.d. of x
        sig_x_vec = np.zeros((approx_horizon,))
        sig_x_vec[0] = self.sig_x0
        for t in range(approx_horizon-1):
            sig_x_vec[t+1] = np.sqrt(self.rho**2 * sig_x_vec[t]**2 + self.sig**2)
        # Prepare list of gird points
        x_max_vec  = Omega * sig_x_vec
        x_grid_list = [
            np.linspace(-x_max_vec[t], x_max_vec[t], N) for t in range(approx_horizon)
        ]
        # Calculate the step sizes
        h = (2 * x_max_vec)/(N-1)
        # Prepare the empty list where the transition matrices will be stored
        trans_mat_list = []
        for t in range(approx_horizon - 1):
            trans_mat_t = [
                [tauchen_trans_mat_ij(i, j, x_grid_list[t+1], x_grid_list[t], h[t+1]) for j in range(N)]
                for i in range(N)
                ]
            trans_mat_list.append(trans_mat_t)
        # Store the result as the instance's attributes
        self.__dict__['{0}_grid_list'.format(self.varname)] = x_grid_list
        self.trans_mat_list, self.step_size = trans_mat_list, h
    
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
        # Expand the transition matrix for the given N
        for n in range(3, N+1, 1):
            Pi_pre = np.copy(Pi_N)
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
            sig_x_vec[t+1] = np.sqrt(self.rho**2 * sig_x_vec[t]**2 + self.sig**2)
        # Prepare list of gird points
        x_max_vec  = sig_x_vec * np.sqrt(N - 1)
        x_grid_list = [
            np.linspace(-x_max_vec[t], x_max_vec[t], N) for t in range(approx_horizon)
        ]
        # Calculate the step sizes
        h = (2 * x_max_vec)/(N-1)
        # Prepare the empty list where the transition matrices will be stored
        trans_mat_list = []
        # approximate transition matirix
        for t in range(approx_horizon - 1):
            # transition probability satisfying the moment condition on the conditional expectation
            pi_t = 0.5 * (1 + self.rho * sig_x_vec[t]/sig_x_vec[t+1])
            # starting from N = 2
            Pi_N = np.array([[    pi_t, 1 - pi_t],
                             [1 - pi_t,     pi_t]])
            # Recursively expand N
            for n in range(3, N+1, 1):
                Pi_pre = np.copy(Pi_N)
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
