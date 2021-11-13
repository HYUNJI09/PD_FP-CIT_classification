import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# bad version
# def StandardScalingData(data, keep_dim=True):
#     #
#     data = np.array(data)
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform([d.flatten() for d in data]) # (N, NF)
#     if keep_dim is True:
#         return scaled_data.reshape(data.shape)
#     else:
#         return scaled_data


def StandardScalingData(data, save_path=None, keep_dim=True, train=True, scaler=None):
    data = np.array(data).astype(np.float32)
    flattened_data = [d.flatten() for d in data]
    if train is True:
        scaler = StandardScaler()
        scaler.fit(flattened_data)
        scaled_data = scaler.transform(flattened_data)
        if save_path :
            save_sk_model(scaler, save_path)
    else :
        if save_path :
            scaler = load_sk_model(save_path)
        scaled_data = scaler.transform(flattened_data)

    print("scaled_data", scaled_data.shape)

    if keep_dim is True:
        return scaler, scaled_data.reshape(data.shape)
    else:
        return scaler, scaled_data

def save_sk_model(model, save_path):
    try:
        joblib.dump(model, save_path)
    except FileNotFoundError as fnfe:
        print(fnfe)
        return False
    return True

def load_sk_model(save_path):
    model = joblib.load(save_path)
    return model

# change the distribution of the input images is distributed over [0, 255] into [-1, 1]
def apply_simple_scaler_for_each_imgs(imgs):
    imgs = np.array(imgs)
    imgs = (imgs-127.5)/127.5
    return imgs

def apply_post_process_for_simple_scaler(imgs, option=None):
    if option == "(0,255)":
        imgs = imgs * 127.5 + 127.5
        #imgs = np.clip(imgs, 0, 255)
    if option == "(0,1)":
        imgs = imgs * 0.5 + 0.5
        #imgs = np.clip(imgs, 0, 255)
    return imgs

def apply_post_process_for_standardscaler(imgs, save_path):
    try:
        model = load_sk_model(save_path)
    except FileNotFoundError as fnfe:
        print(fnfe)
        return False
    imgs = model.inverse_transform(imgs)
    return imgs

# Z-score
def apply_z_score_for_each_imgs(imgs):
    imgs = np.array(imgs)
    n_imgs = len(imgs.shape)
    imgs_shape = imgs.shape
    if n_imgs == 3: # (B, H, W)
        IMG_N, IMG_HEIGHT, IMG_WIDTH = imgs_shape
    elif n_imgs == 4: # (B, H, W, C)
        IMG_N, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL = imgs_shape
    imgs = np.reshape(imgs, (len(imgs), -1))
    imgs_t = np.transpose(imgs, (1, 0))
    imgs_t_norm = (imgs_t - np.mean(imgs_t, axis=0))/np.std(imgs_t, axis=0)
    imgs = np.transpose(imgs_t_norm, (1, 0))
    imgs = np.reshape(imgs, imgs_shape)
    return imgs

def apply_post_process_for_z_score(imgs):
    imgs = np.array(imgs).astype(np.float32)
    res = []
    for img_ in imgs:
        min_ = img_.min()
        #print("debug", img_)
        img_ = img_-min_
        #print("debug",img_.min(), img_.max())
        res.append(img_/img_.max())
    return np.array(res)


# mean_subtraction
# def apply_mean_subtraction(imgs): # same with StandardScalar
#     imgs = np.array(imgs)
#     n_imgs = len(imgs.shape)
#     imgs_shape = imgs.shape
#     if n_imgs == 3: # (B, H, W)
#         IMG_N, IMG_HEIGHT, IMG_WIDTH = imgs_shape
#     elif n_imgs == 4: # (B, H, W, C)
#         IMG_N, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL = imgs_shape
#     imgs = np.reshape(imgs, (len(imgs), -1))
#     imgs_norm = (imgs - np.mean(imgs, axis=0))/np.std(imgs, axis=0)
#     imgs = np.reshape(imgs_norm, imgs_shape)
#     return imgs

# mean_subtraction
def apply_mean_subtraction(imgs): # same with StandardScalar
    imgs = np.array(imgs)
    return (imgs-imgs.mean())/imgs.std()

# min_max_normalization
def apply_min_max_normalization(imgs):
    """
    https://www.oreilly.com/library/view/regression-analysis-with/9781788627306/6bb0d820-6200-4bfe-aa91-e7b7ffa2a9c1.xhtml
    :param imgs:
    :return:
    """
    imgs = np.array(imgs)
    len_shape = len(imgs.shape)
    flat_imgs = np.array([imgs_elem.flatten() for imgs_elem in imgs])

    mins = np.amin(imgs, axis=tuple(range(len_shape))[1:])
    copied_mins = np.array([mins] * flat_imgs.shape[1])
    copied_mins = copied_mins.transpose()

    maxs = np.amax(imgs, axis=tuple(range(len_shape))[1:])
    copied_maxs = np.array([maxs] * flat_imgs.shape[1])
    copied_maxs = copied_maxs.transpose()

    results = (flat_imgs - copied_mins) / (copied_maxs - copied_mins)
    return results.reshape(imgs.shape)