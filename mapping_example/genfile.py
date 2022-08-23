import numpy as np

# The values mac* in the mapping "mac -> mac*" are exported from the SPICE simulation. For indexing the values in the mapping, the actual mac (quantized to interger) value is used. As an example, we here create a dummy mappings, which are loaded into the PyTorch simulation.

# Case 1) In the case of no variation, the mapping is a simple look up table.

# Case 2) In the case with variation, the mapping for each mac value in "mac -> mac*" consists of parameters (mean and standard deviation) for a normal distribution.

manual_mapping = 1

if manual_mapping is not None:
    ### CASE 1

    # old mapping
    manual_mapping_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 14, 14, 17, 17, 17, 21, 21, 21, 21, 26, 26, 26, 26, 26, 26, 26, 31, 31, 31]

    # new mapping
    # manual_mapping_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 28, 30, 30, 32, 32]

    mapping_list_np = np.array(manual_mapping_list, dtype=float)

    print("numpy list", mapping_list_np)

    # save mapping as "mapping.npy"
    with open('mapping.npy', 'wb') as mp:
        np.save(mp, mapping_list_np)
else:
    array_size = 24
    constant_shift = 1
    # create list
    mapping_list = [i for i in range(0,array_size+1)]

    print("Standard mapping", mapping_list)

    # shift entries in list
    mapping_list = [(i+constant_shift) for i in mapping_list]

    print("Shifted mapping", mapping_list)

    # clip values to largest
    for idx in range(len(mapping_list)):
        if mapping_list[idx] > array_size:
            mapping_list[idx] = array_size

    print("Clipped mapping", mapping_list)

    # convert python array to numpy array
    mapping_list_np = np.array(mapping_list, dtype=float)

    print("numpy list", mapping_list_np)

    # save mapping as "mapping.npy"
    with open('mapping.npy', 'wb') as mp:
        np.save(mp, mapping_list_np)

    ### CASE 2
