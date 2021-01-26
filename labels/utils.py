def invert_dict(d):
    return { v:k for k,v in d.items() }

# the nice union operator was introduced in python 3.9. Let's be compatible with older versions too
def merge_dict(a,b):
    return { **a, **b }