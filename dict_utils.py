import os
import pickle

def save_dict(obj, save_name):
    dirname = os.path.dirname(save_name)
    basename = os.path.basename(save_name)
    with open(os.path.join(dirname, basename+'.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(save_name):
    dirname = os.path.dirname(save_name)
    basename = os.path.basename(save_name)
    with open(os.path.join(dirname, basename+'.pkl'), 'rb') as f:
        return pickle.load(f)