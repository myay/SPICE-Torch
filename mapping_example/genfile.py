import numpy as np

# The values mac* in the mapping "mac -> mac*" are exported from the SPICE simulation. For indexing the values in the mapping, the actual mac (quantized to interger) value is used. As an example, we here create a dummy mappings, which are loaded into the PyTorch simulation.

# Case 1) In the case of no variation, the mapping is a simple look up table.

# Case 2) In the case with variation, the mapping for each mac value in "mac -> mac*" consists of parameters (mean and standard deviation) for a normal distribution.

### CASE 1
array_size = 16
constant_shift = 2

# create list
mapping_list = [i for i in range(0,17)]

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
mapping_list_np = np.array(mapping_list)

print("numpy list", mapping_list_np)

# save mapping as "mapping.npy"
with open('mapping.npy', 'wb') as mp:
    np.save(mp, mapping_list_np)

### CASE 2
