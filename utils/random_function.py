import random
import sympy as sp
import numpy as np

import numpy as np

def lin_indep_sample(d, D):
    if d > D:
        raise ValueError("d cannot be greater than D for linear independence.")
    
    while True:
        # Generate a random matrix of shape (d, D)
        random_matrix = np.random.rand(d, D)
        # Check the rank of the matrix
        if np.linalg.matrix_rank(random_matrix) == d:
            return random_matrix

def qr_basis(vectors):
    """
    Find an orthonormal basis using QR decomposition.

    Parameters:
        vectors (ndarray): A (d, D) array of d linearly independent vectors in R^D.

    Returns:
        orthonormal_basis (ndarray): A (d, D) array of orthonormal basis vectors.
    """
    q, _ = np.linalg.qr(vectors.T)  # QR decomposition
    return q.T 

def generate_random_equation(n):
    
    # Create n symbolic variables x1, x2, ..., xn
    variables = sp.symbols(f'x1:{n+1}')
    
    # Define possible operations
    operations = ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log']
    
    # Start building the equation
    equation = 0
    
    # Randomly combine variables with operations
    for _ in range(random.randint(3, 7)):  # Random number of terms in the equation
        var = random.choice(variables)
        op = random.choice(operations)
        
        if op in ['+', '-', '*']:
            coeff = random.uniform(-5, 5)
            equation += coeff * var
        elif op == '/':
            # Add a small constant to avoid division by zero
            coeff = random.uniform(-5, 5)
            equation += coeff / (var + random.uniform(0.1, 1))
        elif op == '**':
            power = random.randint(2, 3)
            equation += var**power
        elif op in ['sin', 'cos', 'exp', 'log']:
            if op == 'log':  # Avoid log of negative or zero
                equation += sp.log(var + random.uniform(1, 5))
            else:
                func = getattr(sp, op)
                equation += func(var)
    
    # Convert to a Python function
    equation_func = sp.lambdify(variables, equation, modules='numpy')
    
    return equation, equation_func

def random_function(dim_total, dim_effect):
    onb = qr_basis(lin_indep_sample(dim_effect, dim_total))
    equation, equation_func= generate_random_equation(dim_effect)
    def func(vector):
        embedding = onb @ vector
        print(*embedding)
        return equation_func(*embedding)
    return func
