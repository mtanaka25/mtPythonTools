import numpy as np

def find_nearest_idx(x, array) -> int:
    """ Find the closest value in array to x and return its index.
    
    Args:
        x (scalar): search value
        array (array-like): array of candidate values
    
    Returns:
        Int : the index for the element closest to x in array
    """
    if np.isscalar(x):
        nearest_idx = np.abs(array - x).argmin()
    else:
        nearest_idx = find_nearest_idx_vec(x, array)
    return nearest_idx

def find_nearest_idx_vec(x_vec, array) -> np.ndarray:
    """ Find the closest values in array to x and return their indeces.
    
    Args:
        x_vec (array-like): vector of search values
        array (array-like): array of candidate values
    
    Returns:
        NumPy array: vector of the index for the element closest to each x in array
    """
    
    if x_vec == np.ndarray:
        x_vec = x_vec.flatten()
    nearest_index_vec = [find_nearest_idx(x_vec_i, array) for x_vec_i in x_vec]
    nearest_index_vec = np.array(nearest_index_vec)
    return nearest_index_vec