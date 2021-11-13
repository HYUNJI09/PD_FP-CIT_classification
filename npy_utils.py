import os
import numpy as np

def npy_mat_saver(mat, save_path):
    mat = np.array(mat)
    dirname = os.path.dirname(save_path)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    # filename= os.path.basename(save_path)
    #save_path = os.path.join(save_dir, save_filename)
    np.save(save_path, mat)
    return
