# Change the working directory to the directory containing the codes
import os
import numpy as np
import dill
from copy import deepcopy
import glob
import gc

path_save =  "/scratch/up6/ai2733/outputs_CASE1/budget/grid_tracking_cropped_new/"
n_slices  = len(glob.glob(f"{path_save}/grid_*"))

def merge_multiple_grids(grids):
    # Create a new instance based on the first grid
    new_instance = deepcopy(grids[0])

    # Iterate over all other grids
    for idx_grid, grid in enumerate(grids[1:]):
        # Merge attributes
        for attr, value in vars(grid).items():
            if hasattr(new_instance, attr):
                current_attr_value = getattr(new_instance, attr)
                
                # The arrays
                if isinstance(current_attr_value, np.ndarray) and isinstance(value, np.ndarray):
                    if attr == "phb":
                        # Always concatenate for 'phb': base-state geopotential, which doesn't vary with time.
                        concatenated_array = np.concatenate((current_attr_value, value), axis=-1)
                        setattr(new_instance, attr, concatenated_array)
                        print (attr, "concatenated")
                    elif not np.array_equal(current_attr_value, value):
                        # Concatenate if not the same as the first grid
                        concatenated_array = np.concatenate((current_attr_value, value), axis=-1)
                        setattr(new_instance, attr, concatenated_array)
                        print (attr, "concatenated")
                    else:
                        # If arrays are the same as the first grid, do nothing
                        print (attr, "is the same - do not concatenate")
                        
                # Time steps
                elif attr == "nt":
                    # Sum the nt attributes
                    new_instance.nt += value
                    print (attr, "summed up")
                    
                # Single values
                else:
                    # For non-array attributes, just set the value from the current grid
                    #setattr(new_instance, attr, value)
                    print (attr, "first value kept")

    return new_instance

# To load and merge one grid at a time
def load_and_merge_grids(file_paths):
    merged_grid = None
    for idx_grid, file_path in enumerate(file_paths):
        with open(file_path, 'rb') as f:
            grid = dill.load(f)
            if merged_grid is None:
                merged_grid = grid  # First grid
            else:
                merged_grid = merge_multiple_grids([merged_grid, grid])
            del grid  # Explicitly delete to free memory
            gc.collect()
        print (f"--------------------------------GRID {idx_grid} MERGED ------------------------")
    return merged_grid

# Load all grids, merge them and save the final grid
file_paths = [f'{path_save}/grid_{idx}.pkl' for idx in range(n_slices)]
merged_grid = load_and_merge_grids(file_paths)

with open(f'{path_save}/All_grid.pkl', 'wb') as f:
    dill.dump(merged_grid, f)
