from skimage.util import view_as_blocks
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

checkpoint_path = "data/model.keras"
model = tf.keras.models.load_model(checkpoint_path)

R = 0
G = 1
B = 2

def gaussian_fn(x, mu, sigma):
    return (np.exp(-((x-mu)**2)/(2.0*sigma*sigma)))

def gaussian(x, depth, k_d, k_c_off):
    ks = 1.5
    km = 3.5
    sigma = (np.mean(k_d)**2)/ks
    mu = np.sqrt(k_c_off[(x*2.0-1.0).astype(int)].sum(-1))*km/(1+(depth/2))
    return gaussian_fn(x, mu, sigma)

def predict_cb(a, x, kc_off_profile):
    x[0][-2] = a[0]
    x[0][-1] = a[1]
    y = model.predict([x], verbose=0)
    return -0.65*y.mean()-gaussian(a[1]*3.0, x[0][9]*5.0, x[0][:3], kc_off_profile)*1.0

def predict(path):
    im = plt.imread(f"{path}.tif").astype(int)
    params = np.loadtxt(f"{path}.txt")

    blocks = view_as_blocks(im, block_shape=(120, 128, 3))
    im_stds = []
    for patches in blocks:
        for patch in patches:
            im_stds.append(np.asarray([patch[0, :, :, R].std(), patch[0, :, :, G].std(), patch[0, :, :, B].std()]))
    stds = np.mean(im_stds, 0)
    means = np.mean(im, (0,1))

    k_d = params[:3]
    k_c_on = params[3:6]
    k_c_off = params[6:9]
    depth = params[9]
    dist = params[10]
    lights = params[17]
    kc_off_profile = np.loadtxt(f"{path}_kc_off.txt")

    ### NORMALISE ###
    dist = dist / 3.0
    depth /= 5.0
    lights = lights / 100.0
    means = means / 255.0
    stds = stds / 15.0
    k_c_on /= 15.0
    k_c_off /= 15.0
    kc_off_profile /= 15.0

    #                                                                         new_l, new_d
    X = np.array([[*k_d, *k_c_on, *k_c_off, depth, dist, *means, *stds, lights, 0.5, 0.5]])
    bounds = [
        (0.0, 1.0),
        (0.5/3.0, 1.0)
    ]
    a0 = np.mean(bounds, -1)
    res = minimize(predict_cb, a0, method='Nelder-Mead', bounds=bounds, args=(X, kc_off_profile),
                options={'disp': False})
    d_lights = res.x[0]*100.0
    d_dist = res.x[1]*3.0
    print(f"\n... Predicting for {path} ...")
    print(f"old distance: {dist*3.0:.2f}m")
    print(f"old light intensity: {lights*100.0:.2f}%")
    print(f"Suggested distance: {d_dist:.2f}m")
    print(f"Suggested lights: {d_lights:.2f}%")

def main():
    set_prefix = "data"
    imgs = ("img_1", "img_2")
    for im in imgs:
        predict(f"{set_prefix}/{im}")

if __name__ == '__main__':
    main()