# %%
import numpy as np

# %%
def non_diagonal_sum(matrix):
    remove_diagonal_array = np.identity(len(matrix)) == False
    return np.sum(np.abs(matrix) * remove_diagonal_array)

# %%
def calculate_variables(a, p, q):
    a_pp, a_pq, a_qq = a[p,p], a[p,q], a[q,q]

    zero_threshold = 0.2 * (non_diagonal_sum(a)/(len(a)**2))

    if(abs(a_pq) > zero_threshold):
        theta = (a_qq - a_pp)/(2 * a_pq)
        t = np.sign(theta)/(abs(theta) + np.sqrt(theta ** 2 + 1))
        c = 1/np.sqrt(t ** 2 + 1)
        s = t * c

        return c, s
    else:
        return 1, 0

# %%
def rotate(matrix, p, q):
    c, s =calculate_variables(matrix, p, q)
    rotation_matrix = np.identity(len(matrix))
    rotation_matrix[p,p], rotation_matrix[q,q] = c, c
    rotation_matrix[p,q] = s
    rotation_matrix[q,p] = -s
    
    return rotation_matrix.T @ matrix @ rotation_matrix

# %%
def jacobi_sweep(matrix):
    active_matrix = matrix.copy()
    for p in range(1, len(active_matrix)):
        for q in range(p):
            active_matrix = rotate(active_matrix, p, q)
    return active_matrix

# %%
def jacobi_eigenvalues(matrix, decimals = 8):
    active_matrix = matrix.copy()
    for i in range(10):
        active_matrix = jacobi_sweep(active_matrix)
        if(non_diagonal_sum(active_matrix) < 10 ** -decimals):
            break
    return np.sort(active_matrix.diagonal())[::-1]

# %%
def eigenvalues_from_unique(the_six, decimals = 8):
    if len(the_six) == 6:
        xx, xy, yy, yz, zz, xz = the_six
        input_matrix = np.array([
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz]
        ])
    
        #print(input_matrix)
    
        eigenvalues = jacobi_eigenvalues(input_matrix, decimals = decimals)
        return eigenvalues
    else:
        print(f'Jacobi: The chosen axis must be of length 6, but instead, got: {the_six}')
        return np.zeros(3)

