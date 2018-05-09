def dot_product(x, w):
    result = 0

    for k in all_keys(x, w):
        result += x.get(k, 0) + w.get(k, 0)

    return result


def vector_scalar_product(vector, scalar):
    product_vector = {}
    for k in all_keys(vector):
        product_vector[k] = vector.get(k, 0) * scalar

    return product_vector


def vectors_sum(v1, v2):
    sum_vector = {}
    for k in all_keys(v1, v2):
        sum_vector[k] = sum_vector.get(k, 0) + v1.get(k, 0) + v2.get(k, 0)

    return sum_vector


def all_keys(v1, v2=None):
    if not v2:
        result = v1.keys()
    else:
        result = list(set(v1) | set(v2))

    return result
