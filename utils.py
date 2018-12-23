def split_list_at_value(list, value):
    new_list = []

    for item in list:
        if item == value:
            new_list.append(item)
            return new_list
        else:
            new_list.append(item)
            
    return new_list

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

# Flatten along first dimension
def flatten(l):
    return [item for sublist in l for item in sublist]