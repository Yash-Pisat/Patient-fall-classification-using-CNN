# import libraries #
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.interpolate import CubicSpline


def add_white_noise(signal, noise_factor):
    """
    Adds white noise to the given signal.
    
    Parameters:
        signal (numpy.ndarray): The original signal.
        noise_factor (float): The magnitude of the noise to add to the signal.
    
    Returns:
        numpy.ndarray: The signal with added white noise.
    """
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_data = signal + noise * noise_factor 
    return augmented_data


def scaling(X, sigma):
    """
    Scale the input data X with a random normal distribution.

    Parameters:
    - X (numpy.ndarray): The input data matrix to be scaled.
    - sigma (float): The standard deviation of the normal distribution used to generate scaling factors.

    Returns:
    - scaled_data (numpy.ndarray): The scaled version of the input data.
    """
    scalingFactor = np.random.normal(loc=1.0, scale= sigma, size= (1,X.shape[0]))
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    scaled_data = X * myNoise[0]
    return scaled_data


def GenerateRandomCurves(X, sigma, knot):
    """
    Generate random curves from an input data.

    Parameters:
    X (numpy array): input data with shape (N,).
    sigma (float): standard deviation used in generating random values.
    knot (int): number of knots in the cubic spline curve.

    Returns:
    numpy array: generated cubic spline curve with shape (N,1).
    """
    xx = (np.ones((X.shape[0],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[0]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    return np.array([cs_x(x_range)]).transpose()

def magnitudewarping(X, sigma, knot):
    """
    Apply magnitude warping to an input data.

    Parameters:
    X (numpy array): input data with shape (N,).
    sigma (float): standard deviation used in generating random values.
    knot (int): number of knots in the cubic spline curve.

    Returns:
    tuple:
        - numpy array: generated cubic spline curve with shape (N,1).
        - numpy array: magnitude warped input data with shape (N,).
    """
    data = X.reshape(X.shape[0],1)
    curve = GenerateRandomCurves(X, sigma, knot)
    mag_warp = data * curve
    mag_warp = mag_warp.reshape(mag_warp.shape[0])
    return curve,mag_warp

# def plotcurve_mw(curve, orig_dummy, aug_dummy , orig_label, aug_label):
#      """
#     Plot curves.

#     Parameters:
#     curve (numpy array): curve data with shape (N,).
#     orig_dummy (numpy array): original dummy data with shape (N,).
#     aug_dummy (numpy array): augmented dummy data with shape (N,).
#     orig_label (str): label for original dummy data.
#     aug_label (str): label for augmented dummy data.
#     """
#     fig,(ax1,ax2,ax3)=plt.subplots(3)
#     ax1.plot(curve,'tab:orange', label ='cubic spline')
#     ax1.axis([0,16500,0,2])
#     l1 = ax1.legend();
#     ax2.plot(orig_dummy,'tab:orange', label = orig_label)
#     ax2.axis([0,16000,600,1500])
#     l2 = ax2.legend();
#     ax3.plot(aug_dummy,'tab:orange', label = aug_label)
#     ax3.axis([0,16000,600,1500])
#     l3 = ax3.legend();

def plotcurve_mw(curve, orig_dummy, aug_dummy , orig_label, aug_label):
    """
    Plot curves.

    Parameters:
    curve (numpy array): curve data with shape (N,).
    orig_dummy (numpy array): original dummy data with shape (N,).
    aug_dummy (numpy array): augmented dummy data with shape (N,).
    orig_label (str): label for original dummy data.
    aug_label (str): label for augmented dummy data.
    """
    fig,(ax1,ax2,ax3)=plt.subplots(3)
    ax1.plot(curve,'tab:orange', label ='cubic spline')
    ax1.axis([0,16500,0,2])
    l1 = ax1.legend();
    ax2.plot(orig_dummy,'tab:orange', label = orig_label)
    ax2.axis([0,16000,600,1500])
    l2 = ax2.legend();
    ax3.plot(aug_dummy,'tab:orange', label = aug_label)
    ax3.axis([0,16000,600,1500])
    l3 = ax3.legend();



def DistortTimesteps(X, sigma, knot):
    """
    This function takes in arrays X, sigma, and knot as input and returns a cumulative graph by adding time intervals obtained from
    GenerateRandomCurves function to the input array X.
    
    """
    tt = GenerateRandomCurves(X, sigma, knot) # Regard these samples around 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,0]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    return tt_cum

def DA_TimeWarp(X, sigma, knot):
    """
    This function takes in arrays X, sigma, and knot as input and returns an interpolated data, reshaped data and original data.The interpolated 
    data is obtained by using the cumulative graph obtained from DistortTimesteps function and the reshaped data is obtained by changing the shape
    of input array X.
    
    """
    tt_new = DistortTimesteps(X, sigma,knot)
    X_new = np.zeros(X.shape)
    X_new = X_new.reshape(X_new.shape[0],1)
    x_range = np.arange(X.shape[0])
    reshaped_x = X.reshape(X.shape[0],1) # changing the shape of dummy data
    X_new[:,0] = np.interp(x_range, tt_new[:,0], reshaped_x[:,0])
    org_x = X_new[:,0]  #Data for STFT plot
    return tt_new,X_new,org_x

def plotcurve_tw(curve,orig_dummy, aug_dummy, orig_label, aug_label):
    """
    This function takes in arrays curve, orig_dummy, aug_dummy, orig_label, and aug_label as input and plots 3 subplots. The first plot shows the
    time cubic spline curve, the second plot shows the original dummy data with the label orig_label and the third plot shows the augmented dummy
    data with the label aug_label.
    
    """
    fig,(ax1,ax2,ax3)=plt.subplots(3)
    ax1.plot(curve,'tab:orange', label ='Time - cubic spline')
    ax1.axis([0,16000,0,16000])
    l1 = ax1.legend();
    ax2.plot(orig_dummy,'tab:orange', label = orig_label)
    ax2.axis([0,16000,600,1800])
    l2 = ax2.legend();
    ax3.plot(aug_dummy,'tab:orange', label = aug_label)
    ax3.axis([0,16000,600,1800])
    l3 = ax3.legend();    


def windowWarp(x, window_ratio=0.1, scales=[0.5, 2.]):
    """
    This function warps a 1D signal by randomly selecting a scale from `scales` and warping a portion of the signal with length proportional to
    `window_ratio`.
    
    Parameters:
    x (np.array): 1D signal to be warped.
    window_ratio (float, optional): Ratio of signal length to be warped, with the default value 0.1.
    scales (list, optional): List of scales to choose from, with the default value [0.5, 2.].
    
    Returns:
    np.array: Warped 1D signal.
    
    """
    warp_scales = np.random.choice(scales)
    warp_size = np.ceil(window_ratio*len(x))
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=len(x)-warp_size-1, size=None)
    window_ends = (window_starts + warp_size).astype(int)
    
    ret = np.zeros_like(x)
    start_seg = x[:window_starts]
    window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales)), window_steps, x[window_starts:window_ends])
    end_seg = x[window_ends:]
    warped = np.concatenate((start_seg, window_seg, end_seg))
    result = np.interp(np.arange(len(x)), np.linspace(0, len(x)-1., num=warped.size), warped).T
    return result



def combined_stft(data_list, label_list):
    """
    Plot the short-time Fourier transform (STFT) of a list of signals with corresponding labels.

    Parameters:
    data_list (list): A list of 1-dimensional signals to be transformed.
    label_list (list): A list of labels, one for each signal in data_list.

    Returns:
    None
    
    """
    
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    for i, ax in enumerate(axs.flat):
        if i >= len(data_list):
            break
        data = data_list[i]
        label = label_list[i]
        f, t, Zxx = stft(data, 1600, nperseg=128)
        f = f[2:]
        t = t[:251]
        Zxx = Zxx[2:,:251]
        ax.pcolormesh(t, f, np.log(np.abs(Zxx)), shading='nearest')
        ax.set_xlabel("t [s]")
        ax.set_ylabel("f [Hz]")
        ax.set_title(label)
    fig.tight_layout()
    plt.show()


def stft_plot(label, data):
    """
    Plots the short-time Fourier transform (STFT) of the given data.

    Parameters
    label: str
        The title for the plot.
    data: array-like
        The input data to be transformed.

    Returns
    None

    Notes
    This function generates a log-scale spectrogram of the data and displays it as a pcolormesh plot. The x-axis is labeled as
    "t [s]" and the y-axis as "f [Hz]".
    
    """
    
    fig, ax = plt.subplots(1, 1, sharex=False, figsize=(5, 3))
    f, t, Zxx = stft(data, 1600, nperseg=128)
    f = f[2:]
    t = t[:251]
    Zxx = Zxx[2:,:251]
    ax.pcolormesh(t, f, np.log(np.abs(Zxx)), shading='nearest')
    ax.set_xlabel("t [s]")
    ax.set_ylabel("f [Hz]")
    ax.set_title(label)
    fig.show()

#--------------Compare 2 STFT plot-------------#
def plot(label1, data1, label2, data2):
    """
    This function plots two spectrogram plots using a shared x-axis.

    Parameters:
    label1 (str): The label for the first plot.
    data1 (np.ndarray): The first data to be plotted.
    label2 (str): The label for the second plot.
    data2 (np.ndarray): The second data to be plotted.

    Returns:
    None
    
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, figsize=(12, 5))
    f1, t1, Zxx1 = stft(data1, 1600, nperseg=128)
    f1 = f1[2:]
    t1 = t1[:251]
    Zxx1 = Zxx1[2:,:251]
    ax1.pcolormesh(t1, f1, np.log(np.abs(Zxx1)), shading='nearest')
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("f [Hz]")
    ax1.set_title(label1)

    f2, t2, Zxx2 = stft(data2, 1600, nperseg=128)
    f2 = f2[2:]
    t2 = t2[:251]
    Zxx2 = Zxx2[2:,:251]
    ax2.pcolormesh(t2, f2, np.log(np.abs(Zxx2)), shading='nearest')
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("f [Hz]")
    ax2.set_title(label2)

    fig.tight_layout()
    fig.show()


def plot_augmented(data, augmented_data, orig_label, aug_label):
    """
    This function plots two graphs on a single figure, one for the original data and another for the augmented data. The two graphs are
    horizontally aligned, sharing the same y-axis, and are of size 12x2. The original data is plotted with orange color and labeled with
    orig_label, while the augmented data is plotted with orange color and labeled with aug_label. Both graphs have their x-axis range set to 
    [0, 16000] and y-axis range set to [600, 1500].
    
    Parameters:
    data (array-like): The original data to be plotted 
    augmented_data (array-like): The augmented data to be plotted
    orig_label (str): The label for the original data
    aug_label (str): The label for the augmented data
    
    Returns:
    None
    
    """
    fig,(ax1,ax2)=plt.subplots(1, 2, sharey = True, figsize=(12,2))
    ax1.plot(data,'tab:orange', label = orig_label)
    ax1.axis([0,16000,600,1500] )
    l1 = ax1.legend();
    ax2.plot(augmented_data,'tab:orange', label = aug_label)
    ax2.axis([0,16000,600,1500])
    l2 = ax2.legend();