import numpy as np
import csv
# The values mac* in the mapping "mac -> mac*" are exported from the SPICE simulation. For indexing the values in the mapping, the actual mac (quantized to interger) value is used. As an example, we here create a dummy mappings, which are loaded into the PyTorch simulation.

# Case 1) In the case of no variation, the mapping is a simple look up table.

# Case 2) In the case with variation, the mapping for each mac value in "mac -> mac*" consists of parameters (mean and standard deviation) for a normal distribution.

manual_mapping = None
load_mapping_csv = 1
load_mapping_path = "32bit_high_res_mappings.csv"
mapping_distr = "32_bit_variation/adc_model_prec_"

if mapping_distr is not None:
    print("Loading distribution-based mapping from CSV")
    # iterate over all confusion matrices
    for i in range(2,33):
        # generate all file names
        list_temp = []
        mapping_to_load = mapping_distr + str(i)
        mapping_to_load += ".csv"
        with open(mapping_to_load, newline='') as csvfile:
            csv_loaded = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(csv_loaded):
                if idx != 0:
                    row_map_np = np.array(row[1:], dtype=float)
                    list_temp.append(row_map_np)
            # create first row with entry 1 at index 0 and value 0 for all other indices
            list_temp_np = np.array(list_temp, dtype=float)
            to_ins = np.array([0 for i in range(0,list_temp_np[0].size)], dtype=float)
            to_ins[0] = 1
            list_temp_np = np.insert(list_temp_np, 0, to_ins, axis=0)
            with open('mapping_cm_{}.npy'.format(i), 'wb') as mp:
                np.save(mp, list_temp_np)
elif load_mapping_csv is not None:
    print("Load from CSV")
    # read csv file
    with open(load_mapping_path, newline='') as csvfile:
        csv_loaded = csv.reader(csvfile, delimiter=',')
        # iterate over rows in csv file
        for idx, row in enumerate(csv_loaded):
            # exclude first row
            if idx != 0:
                # exclude first two values and insert 0 at front of array
                row_map = row[2:]
                row_map.insert(0, 0)
                # convert all entries to integer
                row_map_int = list(map(int, row_map))
                # convert to numpy array and export to file
                row_map_np = np.array(row_map_int, dtype=float)
                with open('mapping_{}.npy'.format(idx), 'wb') as mp:
                    np.save(mp, row_map_np)

elif manual_mapping is not None:
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
