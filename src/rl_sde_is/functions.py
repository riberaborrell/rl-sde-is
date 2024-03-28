
def double_well_1d(x, alpha=1.):
    return alpha * (x**2 - 1) ** 2

def double_well_gradient_1d(x, alpha=1.):
    return 4 * alpha * x * (x**2 - 1)

def double_well_nd(x, alpha):
    return np.sum(alpha * (x**2 - 1) ** 2, axis=1)

def double_well_gradient_nd(x, alpha):
    return 4 * alpha * x * (x**2 - 1)
