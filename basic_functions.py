# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 04:06:34 2023

@author: noga mudrik
"""


"""
Imports
"""
import matplotlib
try:
    from webcolors import name_to_rgb
except:
    pass
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
from numpy.linalg import matrix_power
from scipy.linalg import expm
from math import e
from numpy.core.shape_base import stack
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import random
from pathlib import Path
import os
from tkinter.filedialog import askopenfilename
from datetime import date
#import dill   
import scipy.io
#import mat73
import warnings
import statsmodels as stats
from importlib import reload  
import statsmodels.stats as st
sep = os.sep
from IPython.core.display import display, HTML
from importlib import reload  
from scipy.interpolate import interp1d
#from colormap import rgb2hex
from scipy import interpolate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
try:
    import pylops
except:
    print('did not load pylops')
from statistics import mode
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn import svm
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns




from datetime import datetime as datetime2



sns.set_context('talk')

global ss, today, full_date





full_date = str(datetime2.now())
ss = int(str(datetime2.now()).split('.')[-1])
full_date = full_date.replace('-','_').replace(':', '_').replace('.','_')
today = full_date.split()[0]


"""
parameters
"""
labelpad = 10

labels_sizes = {'title_size': 40, 'xlabel_size': 20,   'ylabel_size': 20    }
label_params = { 'title_params': {'fontsize': labels_sizes.get('title_size', 20)}, 
                'xlabel_params': {'fontsize':  labels_sizes.get('title_size', 20)}, 
                'ylabel_params': {'fontsize':  labels_sizes.get('title_size', 20)}}

from scipy.ndimage import gaussian_filter1d
def calculate_speed_from_pos_vector(pos_x, params={}):
    # Get the Gaussian kernel standard deviation (sigma) from params, or use default value 1.0
    params['sigma'] = params.get('sigma',0.1)
    sigma = params['sigma']
    
    # Apply Gaussian smoothing with the specified sigma
    smoothed_pos_x = gaussian_filter1d(pos_x, sigma)
    # Calculate speed as the difference between consecutive smoothed positions
    speed = np.diff(smoothed_pos_x)
    
    return speed


import json
# def update_nested_dict(d, path, value):
#     # Traverse the dictionary based on the path
#     for key in path[:-1]:
#         d = d.setdefault(key, {})  # Move deeper into the dictionary
#     d[path[-1]] = value  # Set the value at the last key in the path
def open_json_file_utf_8(file_name):
    # Specify the UTF-8 encoding to read the file
    #r"E:\ALL_PHD_MATERIALS\CODES\multi-SiBBlInGS\wikipage_views\CS_education\desktop_spider\langviews-20201009-20241029.json"
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data



from scipy.interpolate import interp1d

def interp_over_time(data, params=None):
    if params is None:
        params = {}
    N, T, p = data.shape
    t = np.arange(T)
    out = np.empty_like(data)
    for i in range(N):
        for j in range(p):
            ts = data[i, :, j]
            mask = ~np.isnan(ts)
            if np.count_nonzero(mask) < 2:
                out[i, :, j] = ts
            else:
                f = interp1d(t[mask], ts[mask], **params)
                out[i, :, j] = f(t)
    return out



    
def array_split_by_duration(ar, durations):
    """
    Splits an array into multiple parts based on specified durations.

    Parameters:
    -----------
    ar : array-like
        The array to be split.
    durations : int, list, tuple, or numpy.ndarray
        Specifies the number of elements in each part:
        - If an integer, the array is divided into equal-sized parts with this length.
        - If a list, tuple, or numpy array, each value represents the size of the corresponding part.

    Returns:
    --------
    list
        A list of subarrays, where each subarray contains elements from the input array based on the specified durations.

    Raises:
    -------
    AssertionError
        If the sum of `durations` does not equal the length of the input array.

    Examples:
    ---------
    >>> import numpy as np
    >>> ar = np.arange(10)
    >>> array_split_by_duration(ar, [3, 4, 3])
    [array([0, 1, 2]), array([3, 4, 5, 6]), array([7, 8, 9])]

    >>> array_split_by_duration(ar, 2)
    [array([0, 1]), array([2, 3]), array([4, 5]), array([6, 7]), array([8, 9])]
    """    
    if not isinstance(durations, (list, tuple, np.ndarray)):
        durations = [durations]*int(len(ar)/durations)
    assert sum(durations) == len(ar), 'sum durations must be equal to length of array but %d vs. %d'%(sum(durations) , len(ar))
    cumsum_durations = np.cumsum([0] + list(durations))
    return [ar[el:el2] for el, el2 in zip(cumsum_durations[:-1], cumsum_durations[1:])]
    


    
def string_to_dict(string):
    try:
        data = json.loads(string)
        # Check if the result is a list of dictionaries and return the first one
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return data[0]
        elif isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass  # Handle error if needed
    return None  # Return None if conversion fails

    
def load_mat_file(mat_name , mat_path = '',sep = sep, squeeze_me = True,simplify_cells = True):
    """
    Function to load mat files. Useful for uploading the c. elegans data. 
    Example:
        load_mat_file('WT_Stim.mat','E:/CoDyS-Python-rep-/other_models')
    """
    if mat_path == '':
        data_dict = sio.loadmat(mat_name, squeeze_me = squeeze_me,simplify_cells = simplify_cells)
    else:
        data_dict = sio.loadmat(mat_path+sep+mat_name, squeeze_me = True,simplify_cells = simplify_cells)
    return data_dict   
def check_if_labels_batches(labels):
    """
    Check if labels are ordered in batches. 
    Checks if the labels form valid batches where no label repeats within a batch.

    This function iterates through pairs of consecutive labels to ensure that no label
    is repeated within a batch. It assumes that the input is a sequence where each
    batch consists of consecutive labels and a valid batch is one where no label 
    appears more than once.

    Parameters:
    labels (list or array-like): A list or array of labels to check for valid batching.

    Returns:
    bool: True if each batch of labels contains unique labels, False otherwise.

    Example:
    >>> check_if_labels_batches(['a', 'b',  'c'])
    True
    >>> check_if_labels_batches(['a', 'b', 'a', 'b'])
    False
    """
    labels_visited = []
    for label1, label2 in zip(labels[:-1], labels[1:]):
        if label1 != label2:
            labels_visited.append(label1)
        if label2 in labels_visited:
            return False
    return True

def plot_3d(mat, params_fig = {}, fig = [], ax = [], params_plot = {}, type_plot = 'plot', to_return = False):
    """
    Plot 3D data.

    Parameters:
    - mat (numpy.ndarray): 3D data to be plotted.
    - params_fig (dict): Additional parameters for creating the figure.
    - fig (matplotlib.figure.Figure): Existing figure to use (optional).
    - ax (numpy.ndarray): Existing 3D subplot axes to use (optional).
    - params_plot (dict): Additional parameters for the plot.
    - type_plot (str): Type of 3D plot ('plot' for line plot, 'scatter' for scatter plot).

    Returns:
    - fig (matplotlib.figure.Figure): The created or existing figure.
    - ax (numpy.ndarray): The created or existing 3D subplot axes.
    """ 
    if checkEmptyList(ax):
        fig, ax = create_3d_ax(1,1, params_fig)
    if type_plot == 'plot':    
        scatter = ax.plot(mat[0], mat[1], mat[2], **params_plot)
    else:
        scatter = ax.scatter(mat[0], mat[1], mat[2], **params_plot)
    if to_return:
        return scatter
    
# def plot_3d(mat, params_fig = {}, fig = [], ax = [], params_plot = {}, type_plot = 'plot', to_return = False):
#     # 
#     if checkEmptyList(ax):
#         fig, ax = create_3d_ax(1,1, params_fig)
#     if type_plot == 'plot':    
#         scatter = ax.plot(mat[0], mat[1], mat[2], **params_plot)
#     else:
#         scatter = ax.scatter(mat[0], mat[1], mat[2], **params_plot)
#     if to_return:
#         return scatter
    

# def plot_3d(mat, params_fig = {}, fig = [], ax = [], params_plot = {}, type_plot = 'plot'):
#     """
#     Plot 3D data.

#     Parameters:
#     - mat (numpy.ndarray): 3D data to be plotted.
#     - params_fig (dict): Additional parameters for creating the figure.
#     - fig (matplotlib.figure.Figure): Existing figure to use (optional).
#     - ax (numpy.ndarray): Existing 3D subplot axes to use (optional).
#     - params_plot (dict): Additional parameters for the plot.
#     - type_plot (str): Type of 3D plot ('plot' for line plot, 'scatter' for scatter plot).

#     Returns:
#     - fig (matplotlib.figure.Figure): The created or existing figure.
#     - ax (numpy.ndarray): The created or existing 3D subplot axes.
#     """
#     if checkEmptyList(ax):
#         fig, ax = create_3d_ax(1,1, params_fig)
#     if type_plot == 'plot':
#         ax.plot(mat[0], mat[1], mat[2], **params_plot)
#     else:
#         ax.scatter(mat[0], mat[1], mat[2], **params_plot)
        
def make_labels_unique_order(labels):
    """
    Returns an array of unique labels, preserving their original order.

    Parameters:
    labels (list or array-like): A list or array of labels which may contain duplicates.

    Returns:
    np.array: A numpy array containing the unique labels in the order they first appear.
    
    Example:
    >>> make_labels_unique_order(['a', 'b', 'a', 'c'])
    array(['a', 'b', 'c'], dtype='<U1')
    """    
    labels_visited = []
    for label in labels:
        if label not in labels_visited:
            labels_visited.append(label)
    return np.array(labels_visited)    


def create_legend(dict_legend, size = 30, save_formats = ['.png','.svg'], 
                  save_addi = 'legend' , dict_legend_marker = {}, 
                  marker = '.', style = 'plot', s = 500, to_save = True, plot_params = {'lw':10}, to_sort_keys = False,
                  save_path = os.getcwd(), params_leg = {}, fig = [], ax = [], figsize = None, to_remove_edges =  True,
                  transparent = True,
                  dict_legend_keys = []):
    
    if len(dict_legend_keys) == 0:
        dict_legend_keys = list(dict_legend.keys())
    if to_sort_keys:
        dict_legend_keys = np.sort(dict_legend_keys)
        
    assert np.array([el in dict_legend for el in dict_legend_keys]).all(), 'pay attention! some elemenets you provided in dict_legend_keys do not exist in dict_legend!'
    if set(dict_legend_keys) != set(list(dict_legend.keys())): print('pay attention! not all keys of dict_legend exist in dict_legend_keys!')
    
    
    if not isinstance(figsize,tuple) and not figsize:
        width = np.max([len(str(el)) for el in dict_legend_keys])
        length = len( dict_legend_keys)
        figsize = (3+(width*size/100)*params_leg.get('ncol', 1) ,3+(length*size/60)/params_leg.get('ncol', 1))
        
    if checkEmptyList(fig) and checkEmptyList(ax):
        fig, ax = plt.subplots(figsize = figsize)    
    else:
        to_remove_edges = False
    
        
    if checkEmptyList(fig) != checkEmptyList(ax):    
        raise ValueError('??')
    
    if style == 'plot':
        [ax.plot([],[], 
                 c = dict_legend[area], label = area, marker = dict_legend_marker.get(area), **plot_params) for area in dict_legend_keys]
    else:
        if len(dict_legend_marker) == 0:
            [ax.scatter([],[], s=s,c = dict_legend.get(area), label = area, marker = marker, **plot_params) for area in dict_legend_keys]
        else:
            [ax.scatter([],[], s=s,c = dict_legend[area], label = area, marker = dict_legend_marker.get(area), **plot_params) for area in dict_legend_keys]
    ax.legend(prop = {'size':size},**params_leg)
    
    if to_remove_edges :
        remove_edges(ax, left = False, bottom = False, include_ticks = False)
    fig.tight_layout()
    
    if to_save:
        
        [fig.savefig(save_path + os.sep + 'legend_areas_%s%s'%(save_addi,type_save), transparent=transparent) 
         for type_save in save_formats]
        print('legend saved in %s'%(save_path + os.sep + 'legend_areas_%s.png'%(save_addi)))
        
        
        
        
        
        
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm



        
        
        

def gaussian_convolve(mat, wind = 10, direction = 1, sigma = 1, norm_sum = True, plot_gaussian = False):
    """
    Convolve a 2D matrix with a Gaussian kernel along the specified direction.

    Parameters:
        mat (numpy.ndarray): The 2D input matrix to be convolved with the Gaussian kernel.
        wind (int, optional): The half-size of the Gaussian kernel window. Default is 10.
        direction (int, optional): The direction of convolution. 
            1 for horizontal (along columns), 0 for vertical (along rows). Default is 1.
        sigma (float, optional): The standard deviation of the Gaussian kernel. Default is 1.

    Returns:
        numpy.ndarray: The convolved 2D matrix with the same shape as the input 'mat'.

    Raises:
        ValueError: If 'direction' is not 0 or 1.
    """
    if direction == 1:
        gaussian = gaussian_array(2*wind,sigma)
        if norm_sum:
            gaussian = gaussian / np.sum(gaussian)
        if plot_gaussian:
            plt.figure(); plt.plot(gaussian)
        mat_shape = mat.shape[1]
        T_or = mat.shape[1]
        mat = pad_mat(mat, np.nan, wind)
        return np.vstack( [[ np.nansum(mat[row, t:t+2*wind]*gaussian)                    
                     for t in range(T_or)] 
                   for row in range(mat.shape[0])])
    elif direction == 0:
        return gaussian_convolve(mat.T, wind, direction = 1, sigma = sigma).T
    else:
        raise ValueError('invalid direction')    
        
# def create_simple_cbar(vmin = 0, vmax = 1, cmap ='viridis', 
#                        to_return = False, center = None, 
#                        cbar_kws = {}, aspect = 10, remove_ticks = False,
#                        to_save = False, save_path = os.getcwd(), 
#                        save_name = 'cbar'):
#     fig, axs = plt.subplots()
#     sns.heatmap(np.random.rand(3,3)*np.nan, vmin = vmin, vmax = vmax , cmap = cmap, center = center, cbar_kws=cbar_kws, ax = axs)
#     # Adjust the width of the colorbar
#     cbar = axs.collections[0].colorbar
#     if aspect:
#         cbar.ax.set_aspect(aspect)
#     remove_edges(axs, left = False, bottom = False, include_ticks = False)
#     if remove_ticks:
#         cbar.ax.tick_params(labelsize=0)  # Hide tick labels
#         cbar.ax.xaxis.set_ticks([])  # Remove x-axis ticks
#         cbar.ax.yaxis.set_ticks([])  # Remove y-axis ticks
#     if to_save:
#         save_fig(save_name, fig, save_path)
#     if to_return:
#         return fig
    
    

def create_simple_cbar(vmin = 0, vmax = 1, cmap = 'Reds', to_return = False,
                       center = None, cbar_kws = {}, aspect = 10, remove_ticks = False, with_edge = False,
                       fig = [], axs = [], save_path = os.getcwd(), to_save = False, save_addi = 'cbar'):
    if checkEmptyList(axs) and checkEmptyList(fig):
        fig, axs = plt.subplots()
    sns.heatmap(np.random.rand(3,3)*np.nan, vmin = vmin, vmax = vmax , cmap = cmap, center = center, cbar_kws=cbar_kws, ax = axs)
    # Adjust the width of the colorbar
    cbar = axs.collections[0].colorbar
    if aspect:
        cbar.ax.set_aspect(aspect)
    #cbar.ax.set_linewidth(2)
    
    #cbar.outline.set_edgecolor('black')  # This is the correct way to set edge color
    #cbar.outline.set_linewidth(1)  # Optionally set the edge width
    if with_edge:
        cbar.ax.spines['top'].set_visible(True)    
        cbar.ax.spines['right'].set_visible(True)
        cbar.ax.spines['bottom'].set_visible(True)
        cbar.ax.spines['left'].set_visible(True) 

    remove_edges(axs, left = False, 
                 bottom = False, include_ticks = False)
    if remove_ticks:
        cbar.ax.tick_params(labelsize=0)  # Hide tick labels
        cbar.ax.xaxis.set_ticks([])  # Remove x-axis ticks
        cbar.ax.yaxis.set_ticks([])  # Remove y-axis ticks
        
    if  with_edge:
        # Set colorbar edge color to black
        cbar.ax.spines['top'].set_edgecolor('black')
        cbar.ax.spines['bottom'].set_edgecolor('black')
        cbar.ax.spines['left'].set_edgecolor('black')
        cbar.ax.spines['right'].set_edgecolor('black')
        
    if to_save:
        save_fig(save_addi, fig, save_path =  save_path)
        
        
    if to_return:
        return fig
        
    
# from SLDS to dLDS
def from_SLDS_z_to_dLDS_c(zs, k):
  # $k$ is the number of sub-dynamics
  # let's assume zs is a list
  # k is the number of discrete states. 
  # first we want to make sure that the states start from 0 and do not skip
  unique_zs = np.unique(zs)
  if len(unique_zs) > k or max(unique_zs) > k - 1:
    raise ValueError('you received more unique discrete states than defined')
  
  arange_states = np.arange(unique_zs.max()+1)

  T = len( zs )
  cs = np.zeros((k, T))

  for t in range(T): #ar_z in arange_states:
    cur_z = zs[t]
    cs[:, t] = 1*(arange_states == cur_z)

  return cs  


from scipy.optimize import linear_sum_assignment
def permute_matrix_to_match(target_matrix, source_matrix):
    """
    Permutes the rows of source_matrix to match target_matrix based on correlation cost.

    Parameters:
    target_matrix (np.ndarray): The target matrix to match.
    source_matrix (np.ndarray): The matrix whose rows are to be permuted.

    Returns:
    np.ndarray: Permuted source_matrix.
    """
    # Compute the correlation matrix (correlation between rows)
    #correlation_matrix = np.corrcoef(target_matrix.T, source_matrix.T) #, rowvar=False)[:target_matrix.shape[0], target_matrix.shape[0]:]
    cost_matrix =np.vstack([[((target_matrix[row] - source_matrix[row2])**2).mean() for row in range(target_matrix.shape[0])] for row2 in range(source_matrix.shape[0])])
    #target_matrix @ source_matrix.T #-correlation_matrix  # Negative correlation as cost

    # Solve the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Permute rows of source_matrix based on the assignment
    permuted_source_matrix = source_matrix[row_indices, :]
     
    # col_indices - how much i need to change the first mat to get the 2nd one
    
    return permuted_source_matrix,col_indices  
# def create_legend(dict_legend, size = 30, save_formats = ['.png','.svg'], 
#                   save_addi = 'legend' , dict_legend_marker = {}, 
#                   marker = '.', style = 'plot', s = 500, to_save = True, plot_params = {'lw':25},
#                   save_path = os.getcwd(), params_leg = {}):
#     fig, ax = plt.subplots()
#     if style == 'plot':
#         [ax.plot([],[], 
#                  c = dict_legend[area], label = area, marker = dict_legend_marker.get(area), **plot_params) for area in dict_legend]
#     else:
#         if len(dict_legend_marker) == 0:
#             [ax.scatter([],[], s=s,c = dict_legend.get(area), label = area, marker = marker, **plot_params) for area in dict_legend]
#         else:
#             [ax.scatter([],[], s=s,c = dict_legend[area], label = area, marker = dict_legend_marker.get(area), **plot_params) for area in dict_legend]
#     ax.legend(prop = {'size':size},**params_leg)
#     remove_edges(ax, left = False, bottom = False, include_ticks = False)
#     fig.tight_layout()
#     if to_save:
#         [fig.savefig(save_path + os.sep + 'legend_areas_%s%s'%(save_addi,type_save)) 
#          for type_save in save_formats]
        
        
        
def str_dict2dict(string):
  """
  Convert a string representation of a dictionary to a Python dictionary.

  Parameters:
  - string (str): String representation of a dictionary.

  Returns:
  - dict: The resulting Python dictionary.
    If the input string is not a valid representation of a dictionary, an empty dictionary is returned.
  """
  string_val = string.replace('"','')
  sub_vals = string_val.replace('{','').replace('}','').split(',')
  sub_sub_vals = [val.split(':') for val in sub_vals]
  #print(sub_sub_vals)
  if np.array([len(el) == 2 for el in sub_sub_vals]).all():
    dict_return  = {val[0].strip():val[1].strip() for val in sub_sub_vals}
    return dict_return
  else:
    return {}


def moving_avg_time(mat, wind = 4):
  """
  Compute the moving average over time for a matrix of spike times.

  Parameters:
  - mat (numpy.ndarray): Binary matrix representation of spike times.
    Rows correspond to neurons, and columns correspond to time bins.
  - wind (int): Size of the moving average window.

  Returns:
  - numpy.ndarray: Matrix of moving averages over time.
    Rows correspond to neurons, and columns correspond to time bins.
  """
  T = mat.shape[1]
  N = mat.shape[0]
  mat = np.hstack([np.zeros((N, wind)), mat, np.zeros((N, wind))])
  return   np.hstack([np.mean(mat[:,i : i + wind] , 1).reshape((-1,1)) for i in range(T)])

def invert_dict(dict_to_invert):
    return {v:k for k,v in dict_to_invert.items()}
    
    
    
def create_scatter_plot(list_times, fig=[], ax=[], res=0.2, to_plot=True, max_time=500, max_neuron=5):
    """
    TODO: merge with "plot_raster"!!
    Create a scatter plot or matrix representation for a list of neuron spike times.

    Parameters:
    - list_times (list of arrays): List containing arrays of spike times for each neuron.
    - fig (matplotlib.figure.Figure): Existing figure to use (optional).
    - ax (matplotlib.axes._subplots.AxesSubplot): Existing axes to use (optional).
    - res (float): Resolution for binning spike times.
    - to_plot (bool): If True, generate and display the scatter plot or matrix; if False, only compute the matrix.
    - max_time (int): Maximum time limit for plotting and matrix computation.
    - max_neuron (int): Maximum number of neurons to include in the analysis.

    Returns:
    - mat (numpy.ndarray): Binary matrix representation of spike times for neurons.
      Rows correspond to neurons, and columns correspond to time bins.
      If 'to_plot' is True, also returns the generated or existing figure and axes.
    """
    list_times = list_times[:max_neuron]

    # Compute minimum and maximum spike times across neurons
    min_val = np.min([np.min(el) for el in list_times if len(el) > 0])
    max_val = np.max([np.max(el) for el in list_times if len(el) > 0])

    # Create figure and axes if not provided and to_plot is True
    if checkEmptyList(ax) and to_plot:
        fig, ax = plt.subplots()

    # Compute the duration based on resolution and limit to max_time
    dur = int(np.ceil((max_val - min_val) / res)) + 1
    dur = np.min([max_time, dur])

    # Initialize the binary matrix for spike times
    mat = np.zeros((len(list_times), dur))

    # Populate the binary matrix with spike times
    for neuron_c, neuron in enumerate(list_times):
        times = ((neuron - min_val) / res).astype(int)
        times = times[times < max_time]
        mat[neuron_c, times] = 1

    # Generate and display the scatter plot or matrix if to_plot is True
    if to_plot:
        sns.heatmap(mat, ax=ax)

    return mat if not to_plot else (mat, fig, ax)



  
def pad_mat(mat, pad_val, size_each = 1, axis = 1):
    if axis == 1:
        each_pad = np.ones((mat.shape[0], size_each))*pad_val
        mat = np.hstack([each_pad, mat, each_pad])
    else:
        each_pad = np.ones((size_each, mat.shape[1]))*pad_val
        mat = np.vstack([each_pad, mat, each_pad])        
    return mat


# def gaussian_convolve(mat, wind = 10, direction = 1, sigma = 1, norm_sum = True, plot_gaussian = False):
#     """
#     Convolve a 2D matrix with a Gaussian kernel along the specified direction.
    
#     Parameters:
#         mat (numpy.ndarray): The 2D input matrix to be convolved with the Gaussian kernel.
#         wind (int, optional): The half-size of the Gaussian kernel window. Default is 10.
#         direction (int, optional): The direction of convolution. 
#             1 for horizontal (along columns), 0 for vertical (along rows). Default is 1.
#         sigma (float, optional): The standard deviation of the Gaussian kernel. Default is 1.
    
#     Returns:
#         numpy.ndarray: The convolved 2D matrix with the same shape as the input 'mat'.
        
#     Raises:
#         ValueError: If 'direction' is not 0 or 1.
#     """
#     if direction == 1:
#         gaussian = gaussian_array(2*wind,sigma)
#         if norm_sum:
#             gaussian = gaussian / np.sum(gaussian)
#         if plot_gaussian:
#             plt.figure(); plt.plot(gaussian)
#         mat_shape = mat.shape[1]
#         T_or = mat.shape[1]
#         mat = pad_mat(mat, np.nan, wind)
#         return np.vstack( [[ np.nansum(mat[row, t:t+2*wind]*gaussian)                    
#                      for t in range(T_or)] 
#                    for row in range(mat.shape[0])])
#     elif direction == 0:
#         return gaussian_convolve(mat.T, wind, direction = 1, sigma = sigma).T
#     else:
#         raise ValueError('invalid direction')


def gaussian_array(length,sigma = 1  ):
    """
    Generate an array of Gaussian values with a given length and standard deviation.
    
    Args:
        length (int): The length of the array.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Default is 1.
    
    Returns:
        ndarray: The array of Gaussian values.
    """
    x = np.linspace(-3, 3, length)  # Adjust the range if needed
    gaussian = np.exp(-(x ** 2) / (2 * sigma ** 2))
    normalized_gaussian = gaussian / np.max(gaussian) # /sum()
    return normalized_gaussian
    
    
    

                 
def create_3d_ax(num_rows, num_cols, params = {}):
    """
    Create a 3D subplot grid.
    
    Parameters:
    - num_rows (int): Number of rows in the subplot grid.
    - num_cols (int): Number of columns in the subplot grid.
    - params (dict): Additional parameters to pass to plt.subplots.
    
    Returns:
    - fig (matplotlib.figure.Figure): The created figure.
    - ax (numpy.ndarray): The created 3D subplot axes.
    """
    fig, ax = plt.subplots(num_rows, num_cols, subplot_kw = {'projection': '3d'}, **params)
    return  fig, ax



def init_mat(size_mat, r_seed = 0, dist_type = 'norm', init_params = {'loc':0,'scale':1}, normalize = False):
  """
  This is an initialization function to initialize matrices. 
  Inputs:
    size_mat    = 2-element tuple or list, describing the shape of the mat
    r_seed      = random seed (should be integer)
    dist_type   = distribution type for initialization; can be 'norm' (normal dist), 'uni' (uniform dist),'inti', 'sparse', 'regional', 'zeros'
    init_params = a dictionary with params for initialization. The keys depends on 'dist_type'.
                  keys for norm -> ['loc','scale']
                  keys for inti and uni -> ['low','high']
                  keys for sparse -> ['k'] -> number of non-zeros in each row
                  keys for regional -> ['k'] -> repeats of the sub-dynamics allocations
    normalize   = whether to normalize the matrix
  Output:
      the random matrix with size 'size_mat'
  """
  np.random.seed(r_seed)
  random.seed(r_seed)
  
  # From normal distribution
  if dist_type == 'norm':
    rand_mat = np.random.normal(loc=init_params['loc'],scale = init_params['scale'], size= size_mat)
    
  # From uniform distribution  
  elif dist_type == 'uni':
    if 'high' not in init_params.keys() or  'low' not in init_params.keys():
        raise KeyError('Initialization did not work since low or high boundries were not set')
    rand_mat = np.random.uniform(init_params['low'],init_params['high'], size= size_mat)
    
  # Initialize w. integers
  elif dist_type == 'inti':
    if 'high' not in init_params.keys() or  'low' not in init_params.keys():
      raise KeyError('Initialization did not work since low or high boundries were not set')
    rand_mat = np.random.randint(init_params['low'],init_params['high'], size= size_mat)
    
  # Init sparse matrix
  elif dist_type == 'sparse':
    if 'k' not in init_params.keys():
      raise KeyError('Initialization did not work since k was not set')
    k=init_params['k']
    b1 = [random.sample(list(np.arange(size_mat[0])),np.random.randint(1,np.min([size_mat[0],k]))) for i in range(size_mat[1])]
    b2 = [[i]*len(el) for i,el in enumerate(b1)]
    rand_mat = np.zeros((size_mat[0], size_mat[1]))
    rand_mat[np.hstack(b1), np.hstack(b2)] = 1
    
  # Localized Spras Initialization
  elif dist_type == 'regional':
    if 'k' not in init_params.keys():
      raise KeyError('Initialization did not work since k was not set for regional initialization')

    k=init_params['k']
    splits = [len(split) for split in np.split(np.arange(size_mat[1]),k)]
    cur_repeats = [np.repeat(np.eye(size_mat[0]), int(np.ceil(split_len/size_mat[0])),axis = 1) for split_len in  splits]
    cur_repeats = np.hstack(cur_repeats)[:size_mat[1]]
    
    rand_mat = cur_repeats
  elif dist_type == 'zeros':
      rand_mat = np.zeros( size_mat)
  else:
    raise NameError('Unknown dist type!')
  if normalize:
    rand_mat = norm_mat(rand_mat)
  return rand_mat

  
def norm_mat(mat, type_norm = 'evals', to_norm = True):
  """
  This function comes to norm matrices by the highest eigen-value
  Inputs:
      mat       = the matrix to norm
      type_norm = what type of normalization to apply. Can be 'evals' (divide by max eval), 'max' (divide by max value), 'exp' (matrix exponential)
      to_norm   = whether to norm or not to.
  Output:  
      the normalized matrix
  """    
  if to_norm:
    if type_norm == 'evals':
      eigenvalues, _ =  linalg.eig(mat)
      mat = mat / np.max(np.abs(eigenvalues))
    if type_norm == 'max':
      mat = mat / np.max(np.abs(mat))
    elif type_norm  == 'exp':
      mat = np.exp(-np.trace(mat))*expm(mat)
  return mat
  
    
def lorenz(x, y, z, s=10, r=25, b=2.667):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def create_lorenz_mat(t = [], initial_conds = (0., 1., 1.05) , txy = []):
  """
  Create the lorenz dynamics
  """
  if len(t) == 0: t = np.arange(0,1000,0.01)
  if len(txy) == 0: txy = t
  # Need one more for the initial values
  xs = np.empty(len(t)-1)
  ys = np.empty(len(t)-1)
  zs = np.empty(len(t)-1)

  # Set initial values
  xs[0], ys[0], zs[0] = initial_conds

  # Step through "time", calculating the partial derivatives at the current point
  # and using them to estimate the next point

  for i in range(len(t[:-2])):
      dt_z = t[i+1] - t[i]
      dt_xy =  txy[i+1] - txy[i]
      x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
      xs[i + 1] = xs[i] + (x_dot * dt_xy)
      ys[i + 1] = ys[i] + (y_dot * dt_xy)
      zs[i + 1] = zs[i] + (z_dot * dt_z)
  return xs, ys, zs


def create_rotation_mat(theta = 0, axes = 'x', dims = 3):
    """
    Create a rotation matrix based on the given parameters.
    
    Parameters:
    theta (float, optional): Angle in radians for rotation. Default is 0.
    axes (str, optional): Axis for rotation. Must be one of 'x', 'y' or 'z'. Default is 'x'.
    dims (int, optional): Dimension of the rotation. Must be either 2 or 3. Default is 3.
    
    Returns:
    numpy.ndarray: Rotation matrix of shape (dims, dims).
    
    Raises:
    ValueError: If dims is not 2 or 3.
    """
    if dims == 3:
        if axes.lower() == 'x':
            rot_mat = np.array([[1,0,0],
                                [0,np.cos(theta), -np.sin(theta)], 
                                [0, np.sin(theta), np.cos(theta)]])
        elif axes.lower() == 'y':
            rot_mat = np.array([[np.cos(theta),0,np.sin(theta)],
                                [0,1, 0], 
                                [-np.sin(theta),0, np.cos(theta)]])
        elif  axes.lower() == 'z':
            rot_mat = np.array([[np.cos(theta),-np.sin(theta),0],
                                [np.sin(theta),np.cos(theta), 0], 
                                [0, 0, 1]])
    elif dims == 2:
        rot_mat = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    else: 
        raise ValueError('dims should be 2 or 3')
    return rot_mat

def flip_power(x1,x2):
    """
    This function takes two arguments, x1 and x2, and returns the result of x2 raised to the power of x1 using the numpy.power function.
    """
    return np.power(x2,x1)

def sigmoid(x, std = 1):
    """
    This function computes the sigmoid function of a given input x, with a standard deviation "std". 
    Parameters
    ----------
    x : np.array / list
    std :  The default is 1.

    Returns
    -------
    np.array
        The sigmoid function maps any input value to the range of 0 and 1, making it useful for binary classification problems and as an activation function in neural networks. 
    """
    return 1 / (1 + np.exp(-x/std))        

def checkEmptyList(obj):
    """
    Parameters
    ----------
    obj : any type

    Returns
    -------
    Boolean variable (whether obj is a list)
    """
    return isinstance(obj, list) and len(obj) == 0
        
def create_dynamics(type_dyn = 'cyl', max_time = 1000, dt = 0.01, change_speed = False, t_speed = np.exp, 
                    axis_speed = [], t_speed_params = {}, to_cent = False,return_3d = False, return_additional = False,
                    params_ex = {}):
  """
  
  Create ground truth dynamics
  dyn_type options:
      cyl
      f_spiral
      df_spiral
      
  """
  params_ex = { **{'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2,'theta':0, 'orientation_ax':'y', 'type_theta':'rotate',
                  'type_theta':'rotate','phi': np.pi/50, 'c_type_for_combined':'sig',
                                'x0':np.array([1,-1]).reshape((-1,1)), 'c_control' :0.999, 'dim':2}, **params_ex}  
  dim = params_ex['dim']
  if t_speed == np.power: 
      t_speed_params = {**{'pow':2}, **t_speed_params}
      t_speed = flip_power
  else:
      t_speed_params = {}

  t = np.arange(0, max_time, dt)
  #if t_speed == np.power: t = t+1
  if type_dyn == 'cyl':
    x = params_ex['radius']*np.sin(t)
    y = params_ex['radius']*np.cos(t)
    z = t     + params_ex['bias']

    if change_speed: 
      t_speed_vec = t_speed(params_ex['exp_power']*t, t_speed_params.get('pow'))
      #print(t)
      #print(t_speed_vec)
      if 0 in axis_speed: x = np.sin(t_speed_vec)
      if 1 in axis_speed: y = np.cos(t_speed_vec)
      if 2 in axis_speed: z = t_speed_vec
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
    
  elif type_dyn == 'comb_spirals':
    x0 = params_ex['x0']
    phi = params_ex['phi']
    c_control_joint = params_ex['c_control']
    if len(x0) == 2: x0 = np.vstack([x0.reshape((-1,1)), [-1]])
    if params_ex['c_type_for_combined'] == 'sig':
        sig_e = 5
        c_control = np.vstack([sigmoid(np.linspace(-sig_e ,sig_e ,len(t)), 2), (1-sigmoid(np.linspace(-sig_e ,sig_e , len(t)), 2))])
        #c_control = c_control/(c_control.sum(0) + 1e-10)
        #c_control = c_control*2
        #c_control /= 100
        #c_control += 0.99
        #plt.figure(); plt.plot(c_control.T)
    else:
        if not isinstance(c_control, (list,np.ndarray, tuple)):
            c_control = [c_control, c_control]
    f1_basis = np.array([[np.cos(phi),np.sin(phi),0],[-np.sin(phi),np.cos(phi),0],[0,0,0.1]])
    f1 = f1_basis.copy() # create_rotation_mat(theta = 0, axes = 'x', dims = 3)# 
    f2 = np.array([[np.cos(phi),0,np.sin(phi)],
                        [0,0.1, 0], 
                        [-np.sin(phi),0, np.cos(phi)]]) #f1_basis.T#create_rotation_mat(theta = np.pi/20, axes = 'y', dims = 3) @ f1_basis#f1[np.array([]),:]#
    # print(f2)
    # x0_copy = x0.copy()
    # for i in range(len(t)):
    #     x0_copy = np.hstack([x0_copy, 0.99*f2 @ x0_copy[:,-1].reshape((-1,1))])    
    # visualize_dyn(x0_copy , remove_back = False)
    for i in range(len(t)):
        cur_mat = (c_control[0,i]*f1 +c_control[1,i]*f2)
        eigenvalues, eigenvectors =  linalg.eig(cur_mat)
        cur_mat = cur_mat / np.max(np.abs(eigenvalues))
        c_control[:,i] = c_control[:,i]/ np.max(np.abs(eigenvalues))
        x0 = np.hstack([x0, c_control_joint*cur_mat @ x0[:,-1].reshape((-1,1))])
    x = x0[0,:]
    y = x0[1,:]
    z = x0[2,:] 
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()])
  elif type_dyn == 'spiral' or  type_dyn == 'f_spiral' or  type_dyn == 'df_spiral' :
    """
    spiral = 1d spiral
    f_spiral = flat spiral
    df spiral = spiral in and out
    
    """
 
    x0 = params_ex['x0']
    phi = params_ex['phi']
    c_control = params_ex['c_control']
    
    f1 = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
    for i in range(len(t)-1):
        x0 = np.hstack([x0, c_control*f1 @ x0[:,-1].reshape((-1,1))])
    x = x0[0,:]
    y = x0[1,:]
    if dim == 3 or type_dyn == 'spiral':
        if dim < 3: print('Pay attention! if you want a 2d spiral type "f_spiral" as the dynamic type (not "spiral")')
        if type_dyn == 'spiral':           z = t         
        elif type_dyn == 'f_spiral': z = np.zeros(t.shape)
        elif type_dyn == 'df_spiral': z = np.zeros(t.shape)
    
    if change_speed: 
      t_speed_vec = t_speed(params_ex['exp_power']*t, t_speed_params.get('pow'))
      if 0 in axis_speed: x = t_speed_vec * np.sin(t_speed_vec)
      if 1 in axis_speed: y = t_speed_vec * np.cos(t_speed_vec)
      if 2 in axis_speed and dim == 3: z = t_speed_vec
    if dim == 3:
        dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
    else:
        dynamics = np.vstack([x.flatten(),y.flatten()])
    if type_dyn == 'df_spiral':
        dynamics = np.hstack([dynamics[:,::-1], dynamics])
        if not type_dyn == 'spiral' and not return_3d:
            dynamics = dynamics[:2,:]
  elif type_dyn == 'lorenz':    
    txy = t
    if change_speed: 
      t_speed_vec = t_speed(params_ex['exp_power']*t, t_speed_params.get('pow'))

      if (0 and 1) in axis_speed: txy = t_speed_vec      
      if 2 in axis_speed: txy = t_speed_vec
    x,y,z  = create_lorenz_mat(t, txy = txy)
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
  elif type_dyn == 'torus':
    R=5;    r=1;
    u=np.arange(0,max_time,dt);
    v=np.arange(0,max_time,dt);
    [u,v]=np.meshgrid(u,v);
    x=(R+r*np.cos(v)) @ np.cos(u);
    y=(R+r*np.cos(v)) @ np.sin(u);
    z=r*np.sin(v);
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
  elif type_dyn == 'circ2d':
    x = params_ex['radius']*np.sin(t)
    y = params_ex['radius']*np.cos(t)
    dynamics = np.vstack([x.flatten(),y.flatten()]) 
  elif type_dyn == 'trans':
      dynamics,_,_,_ = create_smooth_trans(max_time = max_time, dt = dt, sig_e = 5)
  elif type_dyn == 'multi_cyl':
    dynamics0_str = create_dynamics('cyl',max_time = max_time ,dt = dt, params_ex = params_ex)
    dynamics0_str = dynamics0_str - dynamics0_str[:,0].reshape((-1,1))
    dynamics0_inv = dynamics0_str[:,::-1]
    dynamics0     = np.hstack([dynamics0_str, dynamics0_inv ])
    
    list_dyns = []
    for dyn_num in range(params_ex['num_cyls']):
        np.random.seed(dyn_num)
        random_trans = np.random.rand(dynamics0.shape[0],dynamics0.shape[0])-0.5
        transformed_dyn = random_trans @ dynamics0
        list_dyns.append(transformed_dyn)
    dynamics = np.hstack(list_dyns)
  elif type_dyn == 'c_elegans':
      mat_c_elegans = load_mat_file('WT_NoStim.mat','E:\CoDyS-Python-rep-\other_models') # 
      dynamics = mat_c_elegans['WT_NoStim']['traces'].T
  elif type_dyn == 'lorenz_2d':
    txy = t
    if change_speed: 
      #t_speed_vec = t_speed(params['exp_power']*t)
      t_speed_vec = t**params_ex['exp_power']
      if (0 and 1) in axis_speed: txy = t_speed_vec      
      if 2 in axis_speed: txy = t_speed_vec
    x,y,z  = create_lorenz_mat(t, txy = txy)
    dynamics = np.vstack([x.flatten(),z.flatten()]) 
  elif type_dyn.lower() == 'fhn':
    v_full, w_full = create_FHN(dt = dt, max_t = max_time, I_ext = 0.5, b = 0.7, a = 0.8 , tau = 20, v0 = -0.5, 
                                w0 = 0, params = {'exp_power' : params_ex['exp_power'], 'change_speed': change_speed}) 
    dynamics = np.vstack([v_full, w_full])
  elif type_dyn.lower()   == 'monkey_trial':
      dynamics = list(np.load('monkey_data_22_5_82_5.npy',allow_pickle = True))
      dynamics =  dynamics[0].T
  elif type_dyn.lower()   == 'monkey_trials':
      dynamics = list(np.load('monkey_data_22_5_82_5.npy',allow_pickle = True))
      dynamics = [dyn.T for dyn in dynamics]  

  elif type_dyn.lower()   == 'eeg_trial': # one 
      """
      EEG data
      """
      dynamics = red_mean(np.load('EEG_trial_control.npy',allow_pickle = True))
      #dynamics =  dynamics[0].T
  elif type_dyn.lower()   == 'eeg_circle': # one circle, one trial
      dynamics = list(np.load('EEG_circle_control.npy',allow_pickle = True))
      dynamics = [red_mean(dyn) for dyn in dynamics]
      #dynamics =  dynamics[0].T
  elif type_dyn.lower()   == 'eeg_circles': # 
      dynamics = list(np.load('EEG_circles_control.npy',allow_pickle = True))
      dynamics = [red_mean(dyn) for dyn in dynamics]
      #dynamics = [dyn.T for dyn in dynamics] 
  elif type_dyn.lower()   == 'eeg_conditions': #
      dynamics = list(np.load('EEG_conditions.npy',allow_pickle = True))
      dynamics = [red_mean(dyn) for dyn in dynamics]
      #dynamics = [dyn.T for dyn in dynamics]  
      
      
  if params_ex['theta'] > 0:
      if params_ex['type_theta'] == 'rotate':
          rot_mat = create_rotation_mat(theta = params_ex['theta'], axes = params_ex['orientation_ax'], dim = dynamics.shape[0])
          dynamics = rot_mat @ dynamics
      elif params_ex['type_theta'] == 'shift':
          if params_ex['orientation_ax'] == 'x':       dynamics =  dynamics+np.array([params_ex['theta'],0, 0]).reshape((-1,1))
          if params_ex['orientation_ax'] == 'y':       dynamics =  dynamics+np.array([0,params_ex['theta'], 0]).reshape((-1,1))
          if params_ex['orientation_ax'] == 'z':       dynamics =  dynamics+np.array([0,0,params_ex['theta']]).reshape((-1,1))
  if to_cent:
      dynamics = dynamics - np.mean(dynamics,1).reshape((-1,1))
      
  if type_dyn == 'comb_spirals' and return_additional:
     return dynamics, f1, f2, c_control
  if (type_dyn == 'f_spiral' or type_dyn == 'df_spiral') and return_additional:
     return  dynamics, f1, c_control
  return    dynamics


def red_mean(mat, axis = 1):
    """
    Subtract the mean of each row or column in a matrix.
    
    Parameters:
    mat (np.ndarray): The input matrix.
    axis (int, optional): The axis along which the mean should be computed. Default is 1 (mean of each row).
    
    Returns:
    np.ndarray: The matrix with each row or column mean subtracted.
    
    """
    if len(mat.shape) <= 2:
        if axis == 0:
            if len(mat.shape) == 1:
                return mat - mat.mean()
            return mat - mat.mean(axis = axis).reshape((1,-1))
        elif axis == 1:
            if not len(mat.shape) >= 2:
                raise ValueError('The mat is 1d. The input matrix must have a first axis.')
            return mat - mat.mean(axis = axis).reshape((-1,1))
    else:
        ones = np.ones((len(mat.shape)))
        ones[axis] = -1
        reshape = tuple(ones)
        return mat - mat.mean(axis = axis).reshape(reshape)
    
def rgb_to_hex(rgb_vec):
    """
    Convert a RGB vector to a hexadecimal color code.

    Parameters:
    rgb_vec (list): A 3-element list of floats representing the red, green, and blue components of the color. The values should be between 0 and 1.
    
    Returns:
    str: The hexadecimal color code as a string.

    Example:
    >>> rgb_to_hex([0.5, 0.2, 0.8])
    '#8033CC'
    """    
    r = rgb_vec[0]; g = rgb_vec[1]; b = rgb_vec[2]
    return rgb2hex(int(255*r), int(255*g), int(255*b))


def quiver_plot(sub_dyn = [], xmin = -5, xmax = 5, ymin = -5, ymax = 5, ax = [], chosen_color = 'red',
                alpha = 0.4, w = 0.02, type_plot = 'quiver', zmin = -5, zmax = 5, cons_color = False,
                return_artist = False,xlabel = 'x',ylabel = 'y',quiver_3d = False,inter=2, projection = [0,1]):
    """
    Plots a quiver or stream plot on the specified axis.
    
    Parameters
    ----------
    sub_dyn: numpy.ndarray, default: []
        The matrix whose eigenvectors need to be plotted. If an empty list is provided, the default sub_dyn will be set to [[0,-1],[1,0]]
    xmin: float, default: -5
        The minimum value for x-axis.
    xmax: float, default: 5
        The maximum value for x-axis.
    ymin: float, default: -5
        The minimum value for y-axis.
    ymax: float, default: 5
        The maximum value for y-axis.
    ax: matplotlib.axes._subplots.AxesSubplot or list, default: []
        The axis on which the quiver or stream plot will be plotted. If a list is provided, a new figure will be created.
    chosen_color: str or list, default: 'red'
        The color of the quiver or stream plot. 
    alpha: float, default: 0.4
        The alpha/transparency value of the quiver or stream plot.
    w: float, default: 0.02
        The width of the arrows in quiver plot.
    type_plot: str, default: 'quiver'
        The type of plot. Can either be 'quiver' or 'streamplot'.
    zmin: float, default: -5
        The minimum value for z-axis (for 3D plots).
    zmax: float, default: 5
        The maximum value for z-axis (for 3D plots).
    cons_color: bool, default: False
        If True, a constant color will be used for the stream plot. If False, the color will be proportional to the magnitude of the matrix.
    return_artist: bool, default: False
        If True, the artist instance is returned.
    xlabel: str, default: 'x'
        Label for x-axis.
    ylabel: str, default: 'y'
        Label for y-axis.
    quiver_3d: bool, default: False
        If True, a 3D quiver plot will be generated.
    inter: float, default: 2
        The step size for the grids in 3D plots.
    projection: list, default: [0,1]
        The indices of the columns in sub_dyn that will be used for plotting.
        
    Returns
    -------
    h: matplotlib.quiver.Quiver or matplotlib.streamplot.Streamplot
        The artist instance, if return_artist is True.
    """
    if sub_dyn.shape[0] > 2: 
        f_proj = sub_dyn[:,projection]
        sub_dyn = f_proj[projection,:]

    
    if len(sub_dyn) == 0:
        sub_dyn =  np.array([[0,-1],[1,0]])

    
    if ymin >= ymax:
        raise ValueError('ymin should be < ymax')
    elif xmin >=xmax:            
        raise ValueError('xmin should be < xmax')
    else:

        if not quiver_3d:
            if isinstance(ax,list) and len(ax) == 0:
                fig, ax = plt.subplots()
            X, Y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin,ymax))

            new_mat = sub_dyn - np.eye(len(sub_dyn))

            U = new_mat[0,:] @ np.vstack([X.flatten(), Y.flatten()])
            V = new_mat[1,:] @ np.vstack([X.flatten(), Y.flatten()])

            if type_plot == 'quiver':
                h = ax.quiver(X,Y,U,V, color = chosen_color, alpha = alpha, width = w)
            elif type_plot == 'streamplot':

                
                x = np.linspace(xmin,xmax,100)
                y = np.linspace(ymin,ymax,100)
                X, Y = np.meshgrid(x, y)
                new_mat = sub_dyn - np.eye(len(sub_dyn))
                U = new_mat[0,:] @ np.vstack([X.flatten(), Y.flatten()])
                V = new_mat[1,:] @ np.vstack([X.flatten(), Y.flatten()])              

                if cons_color:

                    if len(chosen_color[:]) == 3 and isinstance(chosen_color, (list,np.ndarray)): 
                        color_stream = rgb_to_hex(chosen_color)
                    elif isinstance(chosen_color, str) and chosen_color[0] != '#':
                        color_stream = list(name_to_rgb(chosen_color))
                    else:
                        color_stream = chosen_color

                else:
                    new_mat_color = np.abs(new_mat  @ np.vstack([x.flatten(), y.flatten()]))
                    color_stream = new_mat_color.T @ new_mat_color
                try:
                    h = ax.streamplot(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100),U.reshape(X.shape),V.reshape(Y.shape), color = color_stream) #chosen_color
                except:
                    h = ax.streamplot(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100),U.reshape(X.shape),V.reshape(Y.shape), color = chosen_color) #chosen_color
            else:
                raise NameError('Wrong plot name')
        else:
            if isinstance(ax,list) and len(ax) == 0:
                fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
            X, Y , Z = np.meshgrid(np.arange(xmin, xmax,inter), np.arange(ymin,ymax,inter), np.arange(zmin,zmax,inter))

            new_mat = sub_dyn - np.eye(len(sub_dyn))
            U = np.zeros(X.shape); V = np.zeros(X.shape); W = np.zeros(X.shape); 

            for xloc in np.arange(X.shape[0]):
                for yloc in np.arange(X.shape[1]):
                    for zloc in np.arange(X.shape[2]):
                        U[xloc,yloc,zloc] = new_mat[0,:] @ np.array([X[xloc,yloc,zloc] ,Y[xloc,yloc,zloc] ,Z[xloc,yloc,zloc] ]).reshape((-1,1))
                        V[xloc,yloc,zloc] = new_mat[1,:] @ np.array([X[xloc,yloc,zloc] ,Y[xloc,yloc,zloc] ,Z[xloc,yloc,zloc] ]).reshape((-1,1))
                        W[xloc,yloc,zloc] = new_mat[2,:] @ np.array([X[xloc,yloc,zloc] ,Y[xloc,yloc,zloc] ,Z[xloc,yloc,zloc] ]).reshape((-1,1))

            if type_plot == 'quiver':                    
                h = ax.quiver(X,Y,Z,U,V,W, color = chosen_color, alpha = alpha,lw = 1.5, length=0.8, normalize=True,arrow_length_ratio=0.5)#, width = w
                ax.grid(False)
            elif type_plot == 'streamplot':
                raise NameError('streamplot is not accepted for the 3d case')
         
            else:
                raise NameError('Wront plot name')
    if quiver_3d: zlabel ='z'
    else: zlabel = None
 
    add_labels(ax, zlabel = zlabel, xlabel = xlabel, ylabel = ylabel) 
    if return_artist: return h
            
            

  
def movmfunc(func, mat, window = 3, direction = 0, dist = 'uni'):
  """
  moving window with applying the function func on the matrix 'mat' towrads the direction 'direction'
  dist: can be 'uni' (uniform) or 'gaus' (Gaussian)

  Calculates the moving window with the application of the given function `func` on the matrix `mat` in the direction `direction`.
  
  Parameters:
  - func (callable): The function to apply.
  - mat (numpy.ndarray): The matrix to apply the function to.
  - window (int): The size of the moving window. (default: 3)
  - direction (int): The direction to apply the moving window. 0 for row-wise and 1 for column-wise. (default: 0)
  - dist (str): The distribution to use for weighting. Can be 'uni' for uniform or 'gaus' for Gaussian. (default: 'uni')
  
  Returns:
  numpy.ndarray: The result of applying the moving window to `mat`.
  
  Example:
  >>> import numpy as np
  >>> def myfunc(arr, axis=None):
  ...     return np.sum(arr, axis=axis)
  >>> mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
  >>> movmfunc(myfunc, mat, window=3, direction=0, dist='uni')
  array([[ 6.,  9., 12.],
         [15., 18., 21.],
         [ 9., 12., 15.]])
  
  """
  if len(mat.shape) == 1: 
      mat = mat.reshape((-1,1))
      direction = 0
  if np.mod(window,2) == 1:
      addition = int((window-1)/2)
  else:
      addition = int(window/2)
  if direction == 0:
    
    if dist == 'uni':
        mat_wrap = np.vstack([np.nan*np.ones((addition,np.shape(mat)[1])), mat, np.nan*np.ones((addition,np.shape(mat)[1]))])
        movefunc_res = np.vstack([func(mat_wrap[i-addition:i+addition+1,:],axis = direction) for i in range(addition, np.shape(mat_wrap)[0]-addition)])
        
    elif dist == 'gaus':
        mat_wrap = np.vstack([mat[:addition,:][::-1,:], mat, mat[-addition:,:][::-1,:]])
        if np.mod(window,2) == 1:
            wind = np.hstack([np.arange(np.floor(window/2)),np.floor(window/2),np.arange(np.floor(window/2))[::-1] ])+1
            addi = 1
        else:
            wind = np.hstack([np.arange(np.floor(window/2)),np.arange(np.floor(window/2))[::-1] ])+1
            addi = 0
        wind = wind**2
        wind = wind/np.sum(wind)

        movefunc_res = np.vstack([((wind.reshape((1,-1)) @ mat_wrap[i-addition:i+addition+addi,:]).reshape((1,-1)) ) 
                                  for i in range(addition, np.shape(mat_wrap)[0]-addition)])        
  elif direction == 1:
    if dist == 'uni':
        mat_wrap = np.hstack([np.nan*np.ones((np.shape(mat)[0],addition)), mat, np.nan*np.ones((np.shape(mat)[0],addition))])
        movefunc_res = np.hstack([func(mat_wrap[:,i-addition:i+addition+1],axis = direction).reshape((-1,1)) for i in range(addition, np.shape(mat_wrap)[1]-addition)])

    elif dist == 'gaus':
        mat_wrap = np.hstack([mat[:,:addition][:,::-1], mat, mat[:,-addition:][:,::-1]])
        if np.mod(window,2) == 1:
            wind = np.hstack([np.arange(np.floor(window/2)),np.floor(window/2),np.arange(np.floor(window/2))[::-1] ])+1
            addi = 1
        else:
            wind = np.hstack([np.arange(np.floor(window/2)),np.arange(np.floor(window/2))[::-1] ])+1
            addi = 0
        wind = wind**2
        wind = wind/np.sum(wind)

        movefunc_res = np.hstack([(( mat_wrap[:,i-addition:i+addition+addi] @ wind.reshape((-1,1)) ).reshape((-1,1)) ) 
                                  for i in range(addition, np.shape(mat_wrap)[1]-addition)])  
        
  return movefunc_res


def plot_raster(dict_times, fig = [], ax = [], params_plot = {'marker':'|','s':100},
                figsize = None, return_meta = True, max_neuron = 90, max_time = 100,
               to_save = True, save_path = '', save_name = 'raster', xlabel =  'Time',
               ylabel = 'Neuon', reduce_min=False, min_val = np.nan):
    """
    Plots a raster plot of neural spike times.

    Parameters:
    - dict_times (dict): A dictionary where keys are neuron identifiers and values are arrays of spike times.
    - fig (list, optional): Existing figure object to plot on. If empty, a new figure will be created.
    - ax (list, optional): Existing axis object to plot on. If empty, a new axis will be created.
    - params_plot (dict, optional): Parameters for the scatter plot, such as marker style and size. Default is {'marker': '|', 's': 100}.
    - figsize (tuple, optional): Size of the figure. If None, defaults to (max_time*0.4, max_neuron*0.3).
    - return_meta (bool, optional): Whether to return metadata about the plot. Default is True.
    - max_neuron (int, optional): Maximum number of neurons to plot. Default is 90.
    - max_time (int, optional): Maximum time value for x-axis. Default is 100.
    - to_save (bool, optional): Whether to save the figure. Default is True.
    - save_path (str, optional): Path to save the figure. Default is ''.
    - save_name (str, optional): Name of the saved figure file. Default is 'raster'.

    Returns:
    - tuple: If return_meta is True, returns a tuple (xs_vals, ys_vals, keys2old_keys, keys2new_keys) containing:
        - xs_vals: Flattened list of x-values (spike times) for the scatter plot.
        - ys_vals: Flattened list of y-values (neuron identifiers) for the scatter plot.
        - keys2old_keys: Dictionary mapping new keys to original neuron identifiers.
        - keys2new_keys: Dictionary mapping original neuron identifiers to new keys.
        
    example:
        fig, axs = plt.subplots(len(types_with_neg_pos) , num_trials_show, figsize = (50,20), sharex = True, sharey = True)
        [[plot_raster(fire_rate_per_condition[condition]['trial_%d'%trial_show_num]['Neuron'], params_plot = {'lw':1, 'marker' :'|'}, xlabel = 'sec', ylabel = 'Neuron', ax = axs[condition_num, trial_show_num], fig = fig, return_meta = False, remove_min = True)
            for trial_show_num
            in range(num_trials_show)    
        ] 
        for condition_num, condition in enumerate(types_with_neg_pos)
        ]
    """
    params_plot = {**{'marker':'|','s':100}, **params_plot }
    if np.isnan(min_val):   
        min_val = np.min([np.min(value) for value in list(dict_times.values()) if len(value) > 0])
        
    if reduce_min:
        
        dict_times = {key:val - min_val for key, val in dict_times.items()}
    dict_times = {key:val[val < max_time] for c, (key,val) in enumerate(dict_times.items()) if c < max_neuron}
    # first change keys to number
    unique_keys = np.unique(list(dict_times.keys()))
    
    keys2old_keys = {key: old_key for key, old_key in enumerate(unique_keys)}
    keys2new_keys = invert_dict(keys2old_keys)
    
    new_times = {keys2new_keys[key] : time for key,time in dict_times.items()}
    #     if reduce_min:
    #         min_val = np.min([np.min(value) for value in list(new_times.values()) if len(value) > 0])
    #         print('min val')
    #         print(min_val)
    #         new_times = {key:val - min_val for key, val in new_times.items()}

    ys_dict = {key: [key]*len(time) for key,time in new_times.items()}
    

    
    unique_new_keys = np.unique(list(new_times.keys()))
    ys_vals = lists2list([ys_dict[key] for key in unique_new_keys])
    xs_vals = lists2list([new_times[key] for key in unique_new_keys])

    if checkEmptyList(fig) and checkEmptyList(ax):
        if not isinstance(figsize, tuple):
            figsize = (max_time*0.4, max_neuron*0.3)
        fig, ax = plt.subplots(1,1, figsize =figsize)
    elif checkEmptyList(fig) != checkEmptyList(ax):
        raise ValueError('you must provide both fig and ax to plot')
    ax.scatter(xs_vals, ys_vals, **params_plot)
    #ax.set_xlim([0,max_time])
    ax.set_ylim(bottom = -1)
    remove_edges(ax, include_ticks=True, left = True, bottom = True)
    add_labels(ax, xlabel = xlabel, ylabel = ylabel)
    if to_save:
        save_fig(save_name, fig, save_path)
    if return_meta:
        return xs_vals, ys_vals, keys2old_keys, keys2new_keys, min_val
    
  
def add_labels(ax, xlabel='X', ylabel='Y', zlabel='', title='', xlim = None, ylim = None, zlim = None,xticklabels = np.array([None]),
               yticklabels = np.array([None] ), xticks = [], yticks = [], legend = [], ylabel_params = {},zlabel_params = {}, xlabel_params = {},  title_params = {}):
  """
    Add labels, titles, limits, etc. to a figure.
    
    Parameters:
    ax (subplot): The subplot to be edited.
    xlabel (str, optional): The label for the x-axis. Defaults to 'X'.
    ylabel (str, optional): The label for the y-axis. Defaults to 'Y'.
    zlabel (str, optional): The label for the z-axis. Defaults to ''.
    title (str, optional): The title for the plot. Defaults to ''.
    xlim (list or tuple, optional): The limits for the x-axis. Defaults to None.
    ylim (list or tuple, optional): The limits for the y-axis. Defaults to None.
    zlim (list or tuple, optional): The limits for the z-axis. Defaults to None.
    xticklabels (array, optional): The labels for the x-axis tick marks. Defaults to np.array([None]).
    yticklabels (array, optional): The labels for the y-axis tick marks. Defaults to np.array([None]).
    xticks (list, optional): The positions for the x-axis tick marks. Defaults to [].
    yticks (list, optional): The positions for the y-axis tick marks. Defaults to [].
    legend (list, optional): The legend for the plot. Defaults to [].
    ylabel_params (dict, optional): Additional parameters for the y-axis label. Defaults to {}.
    zlabel_params (dict, optional): Additional parameters for the z-axis label. Defaults to {}.
    xlabel_params (dict, optional): Additional parameters for the x-axis label. Defaults to {}.
    title_params (dict, optional): Additional parameters for the title. Defaults to {}.
    
  """
  
  if xlabel != '' and xlabel != None: ax.set_xlabel(xlabel, **xlabel_params)
  if ylabel != '' and ylabel != None:ax.set_ylabel(ylabel, **ylabel_params)
  if zlabel != '' and zlabel != None:ax.set_zlabel(zlabel,**ylabel_params)
  if title != '' and title != None: ax.set_title(title, **title_params)
  if xlim != None: ax.set_xlim(xlim)
  if ylim != None: ax.set_ylim(ylim)
  if zlim != None: ax.set_zlim(zlim)
  
  if (np.array(xticklabels) != None).any(): 
      if len(xticks) == 0: xticks = np.arange(len(xticklabels))
      ax.set_xticks(xticks);
      ax.set_xticklabels(xticklabels);
  if (np.array(yticklabels) != None).any(): 
      if len(yticks) == 0: yticks = np.arange(len(yticklabels)) +0.5
      ax.set_yticks(yticks);
      ax.set_yticklabels(yticklabels);
  if len(legend)       > 0:  ax.legend(legend)

def remove_background(ax, grid = False, axis_off = True):
    """
    Remove the background of a figure.

    Parameters:
    ax (subplot): The subplot to be edited.
    grid (bool, optional): Whether to display grid lines. Defaults to False.
    axis_off (bool, optional): Whether to display axis lines. Defaults to True.
    """       
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    if not grid:
        ax.grid(grid)
    if axis_off:
        ax.set_axis_off()
    

def visualize_dyn(dyn,ax = [], params_plot = {},turn_off_back = False, marker_size = 10, include_line = False, 
                  color_sig = [],cmap = 'cool', return_fig = False, color_by_dominant = False, coefficients =[],
                  figsize = (5,5),colorbar = False, colors = [],vmin = None,vmax = None, color_mix = False, alpha = 0.4,
                  colors_dyns = np.array(['r','g','b','yellow']), add_text = 't ', text_points = [],fontsize_times = 18, 
                  marker = "o",delta_text = 0.5, color_for_0 =None, legend = [],fig = [],return_mappable = False,
                  remove_back = True, edgecolors='none'):
   """
   Plot the multi-dimensional dynamics
   Inputs:
       dyn          = dynamics to plot. Should be a np.array with size k X T
       ax           = the subplot to plot in (optional)
       params_plot  = additional parameters for the plotting (optional). Can include plotting-related keys like xlabel, ylabel, title, etc.
       turn_off_back= disable backgroud of the plot? (optional). Boolean
       marker_size  = marker size of the plot (optional). Integer
       include_line = add a curve to the plot (in addition to the scatter plot). Boolean
       color_sig    = the color signal. if empty and color_by_dominant - color by the dominant dynamics. If empty and not color_by_dominant - color by time.
       cmap         = cmap
       colors       = if not empty -> pre-defined colors for the different sub-dynamics. Otherwise - colors are according to the cmap.
       color_mix    = relevant only if  color_by_dominant. In this case the colors need to be in the form of [r,g,b]
   Output:
       (only if return_fig) -> returns the figure      
      
   """
   if not isinstance(color_sig,list) and not isinstance(color_sig,np.ndarray): color_sig = [color_sig]
   if isinstance(ax,list) and len(ax) == 0:
       if dyn.shape[0] == 3:
           fig, ax = plt.subplots(figsize = figsize, subplot_kw={'projection':'3d'})  
       else:
           fig, ax = plt.subplots(figsize = figsize)  
           
       

   if include_line:
       if dyn.shape[0] == 3:
           ax.plot(dyn[0,:], dyn[1,:], dyn[2,:],alpha = 0.2)
       else:
           ax.plot(dyn[0,:], dyn[1,:], alpha = 0.2)
   if len(legend) > 0:
       [ax.scatter([],[], c = colors_dyns[i], label = legend[i], s = 10) for i in np.arange(len(legend))]
       ax.legend()
      
   if len(color_sig) == 0: 
       color_sig = np.arange(dyn.shape[1])      
   if color_by_dominant and (coefficients.shape[1] == dyn.shape[1]-1 or coefficients.shape[1] == dyn.shape[1]): 
       if color_mix:
           if len(colors) == 0 or not np.shape(colors)[0] == 3: raise ValueError('colors mat should have 3 rows')
           else:

               color_sig = ((np.array(colors)[:,:coefficients.shape[0]] @ np.abs(coefficients))  / np.max(np.abs(coefficients).sum(0).reshape((1,-1)))).T
               color_sig[np.isnan(color_sig) ] = 0.1

               dyn = dyn[:,:-1]
       else:
           
           color_sig_tmp = find_dominant_dyn(coefficients)
           if len(colors_dyns) > 0: 
               color_sig = colors_dyns[color_sig_tmp]
           elif len(color_sig) == 0:  
               color_sig=color_sig_tmp 
           else:        
               color_sig=np.array(color_sig)[color_sig_tmp] 
           if len(color_sig.flatten()) < dyn.shape[1]: dyn = dyn[:,:len(color_sig.flatten())]
           if color_for_0:

               color_sig[np.sum(coefficients,0) == 0] = color_for_0


   if dyn.shape[0] > 2:
       if len(colors) == 0:
           h = ax.scatter(dyn[0,:], dyn[1,:], dyn[2,:], marker = marker, s = marker_size,
                          c= color_sig,cmap = cmap, alpha = alpha,
                          vmin = vmin, vmax = vmax, edgecolors=edgecolors)
       elif isinstance(colors,str):
           h = ax.scatter(dyn[0,:], dyn[1,:], dyn[2,:], marker =marker, s = marker_size,c= colors, alpha = alpha, edgecolors=edgecolors)
       else:
           h = ax.scatter(dyn[0,:], dyn[1,:], dyn[2,:], marker =marker, s = marker_size,c= color_sig, alpha = alpha, edgecolors=edgecolors)
   else:
       dyn = np.array(dyn)
       
       if len(colors) == 0:
           h = ax.scatter(dyn[0,:], dyn[1,:],  marker = marker, s = marker_size,c= color_sig,cmap = cmap, edgecolors=edgecolors, alpha = alpha,
                          vmin = vmin, vmax = vmax)
       elif isinstance(colors,str):
           h = ax.scatter(dyn[0,:], dyn[1,:], marker =marker, s = marker_size,c= colors, edgecolors=edgecolors, alpha = alpha)
       else:
           h = ax.scatter(dyn[0,:], dyn[1,:],  marker = marker, s = marker_size,c= color_sig, alpha = alpha, edgecolors=edgecolors)
  
           params_plot['zlabel'] = None
   if len(params_plot) > 0:
     if dyn.shape[0] == 3:
         if 'xlabel' in params_plot.keys():
           add_labels(ax, xlabel=params_plot.get('xlabel'), ylabel=params_plot.get('ylabel'), zlabel=params_plot.get('zlabel'), title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
         elif 'zlabel' in params_plot.keys():
               add_labels(ax,  zlabel=params_plot.get('zlabel'), title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
         else:
           add_labels(ax,   title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
     else:
         if 'xlabel' in params_plot.keys():
           add_labels(ax, xlabel=params_plot.get('xlabel'), ylabel=params_plot.get('ylabel'), zlabel=None, title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =None)
         elif 'zlabel' in params_plot.keys():
               add_labels(ax,  zlabel=None, title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =None)
         else:
           add_labels(ax,   title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =None,zlabel = None);
   if len(text_points) > 0:
       
       if dyn.shape[0] == 3:
           [ax.text(dyn[0,t]+delta_text,dyn[1,t]+delta_text,dyn[2,t]+delta_text, '%s = %s'%(add_text, str(t)),  fontsize =fontsize_times, fontweight = 'bold') for t in text_points]
       else:
           [ax.text(dyn[0,t]+delta_text,dyn[1,t]+delta_text, '%s = %s'%(add_text, str(t)),  fontsize =fontsize_times, fontweight = 'bold') for t in text_points]
   if remove_back:
       remove_edges(ax)
       ax.set_axis_off()
   if colorbar:
       fig.colorbar(h, cax=ax, position = 'top')
   if return_mappable:
       return h
       
   
    
def find_dominant_row(coefficients):
    """
    This function returns the row index of the largest absolute value of each column in the input 2D numpy array "coefficients".
    
    Inputs:
        coefficients - a 2D numpy array of shape (m, n) where m is the number of rows and n is the number of columns.
        
    Outputs:
        domi - a 1D numpy array of shape (n,) where each element is an integer representing the row index of the largest absolute value of each column.
    """
    domi = np.argmax(np.abs(coefficients),0)
    return domi  


def add_arrow(ax, start, end,arrowprops = {'facecolor' : 'black', 'width':1.8, 'alpha' :0.5} ):
    """
    Add an arrow to the `ax` axis.
    
    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis to add the arrow to.
    start : tuple of floats
        The starting coordinates of the arrow in (x, y) format.
    end : tuple of floats
        The ending coordinates of the arrow in (x, y) format.
    arrowprops : dict, optional
        A dictionary of properties for the arrow, by default 
        {'facecolor': 'black', 'width': 1.8, 'alpha': 0.5}.
        
    Returns
    -------
    None
    
    """    
    arrowprops = {**{'facecolor' : 'black', 'width':1.8, 'alpha' :0.5, 'edgecolor':'none'}, **arrowprops}
    ax.annotate('',ha = 'center', va = 'bottom',  xytext = start,xy =end,
                arrowprops = arrowprops)

  
def plot_3d_color_scatter(latent_dyn,coefficients, ax = [], figsize = (15,10), delta = 0.4, colors = []):
    """
    Plot a 3D color scatter plot.
    
    Parameters
    ----------
    latent_dyn : numpy.ndarray
        A 3xN numpy array representing the latent dynamics.
    coefficients : numpy.ndarray
        A KxN numpy array representing the coefficients.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        A 3D axis to plot on, by default []
    figsize : tuple, optional
        The size of the figure, by default (15, 10)
    delta : float, optional
        The delta between each row, by default 0.4
    colors : list of str, optional
        The colors for each row, by default []
    
    Returns
    -------
    None
    
    """
    if latent_dyn.shape[0] != 3:
        print('Dynamics is not 3d')
        pass
    else:
        if len(colors) == 0:
            colors = ['r','g','b']
        if isinstance(ax,list) and len(ax) == 0:
            fig, ax = plt.subplots(figsize = figsize, subplot_kw={'projection':'3d'})  
        for row in range(coefficients.shape[0]):
            coefficients_row = coefficients[row]
            coefficients_row[coefficients_row == 0]  = 0.01
            
            ax.scatter(latent_dyn[0,:]+delta*row,latent_dyn[1,:]+delta*row,latent_dyn[2,:]+delta, s = coefficients_row**0.3, c = colors[row])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)

def add_dummy_sub_legend(ax, colors,lenf, label_base = 'f'):
    """
    Add a sub-legend to the plot for the specified colors.
    
    Parameters:
    - ax (matplotlib.axes.Axes): The matplotlib axes to add the sub-legend to.
    - colors (list): A list of colors to add to the sub-legend.
    - lenf (int): The number of colors to include in the sub-legend.
    - label_base (str): The base string for the label of each color. (default: 'f')
    
    Returns: None
    """
    dummy_lines = []
    for i,color in enumerate(colors[:lenf]):
        dummy_lines.append(ax.plot([],[],c = color, label = '%s %s'%(label_base, str(i)))[0])    
    legend = ax.legend([dummy_lines[i] for i in range(len(dummy_lines))], ['%s %s'%(label_base,str(i)) for i in range(len(colors))], loc = 'upper left')
    
    ax.legend()
 
def spec_corr(v1,v2, to_abs = True):
  """
  Compute the absolute value of the correlation between two arrays.
  
  Parameters:
  - v1 (numpy.ndarray): The first array to compute the correlation between.
  - v2 (numpy.ndarray): The second array to compute the correlation between.
  - to_abs (bool): Whether to compute the absolute value of the correlation (default: True).
  
  Returns:
  - float: The absolute value of the correlation between `v1` and `v2`.
  """
  corr = np.corrcoef(v1[:],v2[:])
  if to_abs:
      return np.abs(corr[0,1])
  return corr[0,1]

def create_ax(ax, nums = (1,1), size = (10,10), proj = 'd2',return_fig = False,sharey = False, sharex = False, fig = []):
    """
    Create axes in the figure for plotting.

    Parameters:
    ax (list or Axes): List of Axes objects or a single Axes object
    nums (tuple): Number of rows and columns for the subplots (default (1,1))
    size (tuple): Size of the figure (default (10,10))
    proj (str): Projection type ('d2' for 2D or 'd3' for 3D) (default 'd2')
    return_fig (bool): Return the figure object in addition to the Axes object (default False)
    sharey (bool): Share y axis between subplots (default False)
    sharex (bool): Share x axis between subplots (default False)
    fig (Figure): Figure object

    Returns:
    Axes or tuple: The Axes object(s) for plotting
    """
    if isinstance(ax, list) and len(ax) == 0:
        #print('inside')
        if proj == 'd2':
            fig,ax = plt.subplots(nums[0], nums[1], figsize = size, sharey = sharey, sharex = sharex)
        elif proj == 'd3':
            fig,ax = plt.subplots(nums[0], nums[1], figsize = size,subplot_kw={'projection':'3d'}, sharey = sharey, sharex = sharex)
        else:
            raise NameError('Invalid proj input')
        if return_fig:
            return fig, ax

    if  return_fig :
        return fig, ax
    return ax

def nullify_part(f,axis = 'both', percent0 = 80):
    """
    Nullify a part of a matrix.

    Parameters:
    f (numpy array): The input matrix
    axis (str or int): The axis along which to perform the operation ('0', '1', or 'both') (default 'both')
    percent0 (int): The percentile value used to determine which values to nullify (default 80)

    Returns:
    numpy array: The input matrix with the specified values set to 0
    """    
    if not isinstance(axis, str): axis = str(axis)
    if axis == 'both':
        f[f < np.percentile(np.abs(f), percent0)] = 0
    elif axis == '0':
        perc = np.percentile(np.abs(f), percent0, axis = 0)
        for col in range(f.shape[1]):
            f[f[:,col] < perc[col],col] =0
    elif axis == '1':
        perc = np.percentile(np.abs(f), percent0, axis = 1)
        for row in range(f.shape[0]):
            f[row,f[row,:] < perc[row]] =0
    return f

def  create_orth_F(num_subdyns, num_neurons, evals = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], seed_f = 0 , dist_type = 'random' ):
    """
    Create orthogonal matrices.
    
    Parameters:
    num_subdyns (int): Number of sub-dynamics
    num_neurons (int): Number of neurons
    evals (list): List of eigenvalues.
    seed_f (int): Seed for the random number generator (default 0)
    dist_type (str): Distribution type ('random')
    
    Returns:
    list: List of orthogonal matrices
    """
    if num_neurons > len(evals): evals = evals + [0]*(num_neurons - len(evals))
    np.random.seed(seed_f)
    orth_mats = [np.linalg.qr(np.random.rand(num_neurons,num_neurons))[0]
                 for num_subdyn in range(num_neurons)]
    evecs = [np.hstack([orth_mat[:,i].reshape((-1,1)) 
                        for orth_mat in orth_mats]) for i in range(num_neurons)]
    F = [evec @ np.diag(evals[:evec.shape[0]]) @ np.linalg.pinv(evec) for i, evec in enumerate(evecs)]
    np.random.seed(seed_f)
    if len(F)< num_subdyns:
        print('Only %d sud-dyns are  orthogonal')
        if dist_type == 'random' :
            F2 = [np.random.randn(num_neurons,num_neurons) for f_num in range(num_subdyns - len(F) ) ]
        else:
            raise ValueError('Unknown dist type')
        F =  F + F2
    return F[:num_subdyns]
  
def check_save_name(save_name, invalid_signs = '!@#$%^&*.,:;', addi_path = [], sep=sep)  :
    """
    Check if the given file name is valid and returns the final file name.
    The function replaces invalid characters in the file name with underscores ('_').
    
    Parameters:
    save_name (str): The name of the file to be saved.
    invalid_signs (str, optional): A string of invalid characters. Defaults to '!@#$%^&*.,:;'.
    addi_path (list, optional): A list of additional paths to be appended to the file name. Defaults to [].
    sep (str, optional): The separator used between different elements of the path. Defaults to the system separator.
    
    Returns:
    str: The final file name with invalid characters replaced and with additional path appended if provided.

    """
    for invalid_sign in invalid_signs:   save_name = save_name.replace(invalid_sign,'_')
    if len(addi_path) == 0:    return save_name
    else:   
        path_name = sep.join(addi_path)
        return path_name +sep +  save_name

def save_file_dynamics(save_name, folders_names,to_save =[], invalid_signs = '!@#$%^&*.,:;', sep  = sep , type_save = '.npy'):
    """
    Save dynamics & model results to disk.

    Parameters:
    save_name (str): The name of the file to save.
    folders_names (List[str]): List of folder names where the file should be saved.
    to_save (List, optional): List of values to save. Defaults to [].
    invalid_signs (str, optional): String of invalid characters to be removed from the save name. Defaults to '!@#$%^&*.,:;'.
    sep (str, optional): Separator to use when joining `folders_names`. Defaults to `os.sep`.
    type_save (str, optional): The file format to save the data in. Valid options are '.npy' and '.pkl'. Defaults to '.npy'.

    Returns:
    None
    """                    
    save_name = check_save_name(save_name, invalid_signs)
    path_name = os.getcwd() + os.sep + sep.join(folders_names)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    if type_save == '.npy':
        if not save_name.endswith('.npy'): save_name = save_name + '.npy'
        np.save(path_name +sep +  save_name, to_save)
        print('saved in '+ path_name +sep +  save_name)
    elif type_save == '.pkl':
        if not save_name.endswith('.pkl'): save_name = save_name + '.pkl'
        dill.dump_session(path_name +sep +  save_name)
        
        
def find_mid(edges):
    return 0.5*(edges[1:] + edges[:-1])




        
        
        
def save_fig(name_fig,fig, save_path = '', formats = ['png','svg'], save_params = {}, verbose = True) :
    if len(save_path) == 0:
        save_path = os.getcwd()
    if 'transparent' not in save_params:
        save_params['transparent'] = True
    [fig.savefig(save_path + os.sep + '%s.%s'%(name_fig, format_i), **save_params) for format_i in formats]
    if verbose:
        print('saved figure: %s'%(save_path + os.sep + '%s.%s'%(name_fig, 'png')))

def load_pickle(path):
    """
    Load a pickled object from disk.

    Parameters:
    path (str): The path to the pickled object.

    Returns:
    dct (obj): The loaded object.
    """    
    with open(path,'rb') as f:
        dct = pickle.load(f)
    return dct
    

def saveLoad(opt,filename):
    """
    Save or load a global variable 'calc'
    
    Parameters
    ----------
    opt : str
        the option, either "save" or "load"
    filename : str
        the name of the file to save or load from
        
    Returns
    -------
    None
    
    """   
    global calc
    if opt == "save":
        f = open(filename, 'wb')
        pickle.dump(calc, f, 2)
        f.close
     
    elif opt == "load":
        f = open(filename, 'rb')
        calc = pickle.load(f)
    else:
        print('Invalid saveLoad option')
        
def load_vars(folders_names ,  save_name ,sep=sep , ending = '.pkl',full_name = False):
    """
    Load results previously saved.
    
    Parameters:
    folders_names (str/list): List of folders to form the path or a string representation of the path
    save_name (str): Name of the saved file
    sep (str): Separator to join the folders
    ending (str): File extension of the saved file
    full_name (bool): If True, folders_names and sep are ignored
    
    Example:
        load_vars('' ,  'save_c.pkl' ,sep=sep , ending = '.pkl',full_name = False)
    """
    if full_name: 
        dill.load_session(save_name)    
    else:
        if len(folders_names) > 0: path_name = sep.join(folders_names)
        else: path_name = ''
      
        if not save_name.endswith(ending): save_name = '%s%s'%(save_name , ending)
        dill.load_session(path_name +sep +save_name)

   
def str2bool(str_to_change):
    """
    Transform a string representation of a boolean value to a boolean variable.
    
    Parameters:
    str_to_change (str): String representation of a boolean value
    
    Returns:
    bool: Boolean representation of the input string
    
    Example:
        str2bool('true') -> True
    """
    if isinstance(str_to_change, str):
        str_to_change = (str_to_change.lower()  == 'true') or (str_to_change.lower()  == 'yes')  or (str_to_change.lower()  == 't')
    return str_to_change

from glob import glob
def identify_file_within_path(directory, 
                              start_with = '', 
                              end_with = '',
                              within = '',
                              format_file = '.mat', 
                              recursive = False,
                              enable_multiple_files = False):
    """
    Identifies files in a specified directory based on name patterns and file format.

    Parameters:
    - directory (str): Path to the directory where files are searched.
    - start_with (str, optional): Filter for files that start with this prefix. Default is '' (no filter).
    - end_with (str, optional): Filter for files that end with this suffix. Default is '' (no filter).
    - within (str, optional): Filter for files that contain this substring. Default is '' (no filter).
    - format_file (str, optional): File extension to filter (e.g., '.mat'). Default is '.mat'.
    - recursive (bool, optional): If True, searches files recursively in subdirectories. Default is False.
    - enable_multiple_files (bool, optional): If True, allows multiple matching files to be returned. Default is False.

    Returns:
    - str: The matching file name if `enable_multiple_files` is False.
    - list: A list of matching file names if `enable_multiple_files` is True.
    
    Raises:
    - AssertionError: If `enable_multiple_files` is False and multiple files match the criteria.
    """    
    if not format_file.startswith('.'):
        format_file = '.' + format_file
    
    files = glob(directory  + os.sep + '*' + format_file, recursive = recursive)
    print(files)
    files_only = [f for f in files if os.path.isfile(f)]
    file_names = [os.path.basename(directory) for file_path in files_only]
    if len(np.unique(file_names)) != len(np.unique(files_only)):
        print('pay attention, file names are not unique')
    
    file_names_to_include_list = []
    for file_num, file_name in enumerate(file_names):    
        start_with_condition = (start_with and file_name.startswith(start_with)) or (not start_with)
        end_with_condition = (end_with 
                                and file_name.endswith(end_with)) or (not end_with)
        within_condition = (within and file_name.endswith(within)) or (not within)
        if start_with_condition  and end_with_condition  and within_condition:
            file_names_to_include_list.append(file_name)
    if enable_multiple_files or len(file_names_to_include_list) == 0:
        return file_names_to_include_list
    else:
        assert len(file_names_to_include_list) == 1
        return enable_multiple_files[0]
        
    
    
    



def create_readme(text, name="readme.txt", directory=None):
    """Creates a text file with the given name and content in the specified directory."""
    if directory:
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        file_path = os.path.join(directory, name)
    else:
        file_path = name

    with open(file_path, "w") as file:
        file.write(text)

    return file_path  

def find_closest(vec1, vec2, metric = 'mse'):
    """
    Find the closest elements in vec2 for each element in vec1.
    
    Parameters:
    vec1 (ndarray): 1-D numpy array
    vec2 (ndarray): 1-D numpy array
    metric (str): Metric to use for comparison, 'mse' by default
    
    Returns:
    tuple:
        - ndarray: closest elements in vec2 for each element in vec1
        - ndarray: indices of closest elements in vec2 for each element in vec1
    
    Example:
        find_closest([1, 2, 3], [0, 4, 5]) -> ([0, 4, 5], [0, 1, 2])
    """
    if metric == 'mse':
        tiled_vec1 = np.tile(vec1.reshape((1,-1)), [len(vec2),1]) 
        tiled_vec2 = np.tile(vec2.reshape((1,-1)), [len(vec1),1]).T
        v1_closest_to_v2_args = np.argmin((tiled_vec1 - tiled_vec2)**2, 1)
        v1_closest_to_v2 = vec1[v1_closest_to_v2_args]
        return v1_closest_to_v2, v1_closest_to_v2_args
    else:
        raise ValueError('undefined metric!')

def create_colors(len_colors, perm = [0,1,2], style = 'random', cmap  = 'viridis', shuffle_colors = False, shuffle_seed = 0):
    """
    Create a set of discrete colors with a one-directional order
    Input: 
        len_colors = number of different colors needed
    Output:
        3 X len_colors matrix decpiting the colors in the cols
    """
    if style == 'random':
        colors = np.vstack([np.linspace(0,1,len_colors),(1-np.linspace(0,1,len_colors))**2,1-np.linspace(0,1,len_colors)])
        colors = colors[perm, :]
        assert not shuffle_colors, "TB done"
    else:
        
        # Define the colormap you want to use
        #cmap = plt.get_cmap()  # Replace 'viridis' with the desired colormap name

        cmap = plt.get_cmap(cmap) 
        # Create an array of values ranging from 0 to 1 to represent positions in the colormap
        positions = np.linspace(0, 1, len_colors)
        
        # Generate the list of colors by applying the colormap to the positions
        colors = [cmap(pos) for pos in positions]
        
        # You can now use the 'colors' list as a list of colors in your application
        if shuffle_colors:
            random.seed(shuffle_seed)
            random.shuffle(colors)
            
    return colors


def order_grandchildren_in_hierarchical_dict(hierarchical_dict_cur, cur_key = '', cur_list = []):
  # example:
  # #simple_dict = {'parent':{'kid1':{}, 'kid2':{}, 'kid3':{'grand1':{}, 'grand2':{}}}}
  # order_grandchildren_in_hierarchical_dict(simple_dict)
  if len(hierarchical_dict_cur) == 0:
    cur_list.append(cur_key)
    return cur_list
  else:
    for key, val in hierarchical_dict_cur.items():
      next_v = order_grandchildren_in_hierarchical_dict(val, key, cur_list)
  return cur_list



def my_isin(vec, vec_edges = [], return_style = 'all', val_nonzero = 1, num_point = 8, allow_part_time = False):    
    """
    Determine the positions of firings within specified edges an return the result in different formats.

    Parameters:
    vec : array-like
        The times of firing.
    vec_edges : array-like, optional
        Either a list of lists (with 2 elements in each sub-list), an N x 2 vector, or a 1D vector of edges.
    return_style : str, optional
        The style of return value. Options are 'locs' (default), 'array', or 'all'.
    val_nonzero : int, optional
        The value to mark non-zero positions in the return array. Default is 1.
    num_point : int, optional
        The number of points to generate edges if vec_edges is not provided. Default is 8.
    allow_part_time : bool, optional
        If False, raises an error if vec contains values outside the range of vec_edges. Default is False.

    Returns:
    If return_style is 'locs':
        locs : numpy.ndarray
            The locations of firings within the edges.
    If return_style is 'array':
        ar : numpy.ndarray
            An array with non-zero values at firing positions.
    If return_style is 'all':
        locs : numpy.ndarray
            The locations of firings within the edges.
        ar : numpy.ndarray
            An array with non-zero values at firing positions.
        vec_edges : numpy.ndarray
            The edges used for determining the positions.

    Raises:
    ValueError
        If there are more values in vec than in vec_edges and allow_part_time is False.

    Examples:
    >>> vec = np.array([2.54345, 3.434534, 6.54354353])
    >>> my_isin(vec)
    array([0, 1, 0, 0, 0, 1, 0, 0])
    """
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)
        
    if checkEmptyList(vec_edges):
        # create edges   
        
        vec_edges = np.linspace(vec.min(), vec.max() , num_point)
        if len(vec_edges) == 0:
            print('resolution to coarse')
            
    if (isinstance(vec_edges, np.ndarray) and vec_edges.ndim == 1) or (vec_edges.ndim == 2 and (1 in vec_edges.shape)):

        vec_edges = np.hstack([vec_edges[:-1].reshape((-1,1)) , vec_edges[1:].reshape((-1,1)) ])    
   
    if not isinstance(vec_edges, np.ndarray) or  (isinstance(vec_edges, np.ndarray) and vec_edges.ndim == 1):
        vec_edges = np.vstack(vec_edges)
        
    if vec.max() > vec_edges.max() and not allow_part_time:
        raise ValueError('more values in edges then in vec_edges')
   
    ar = np.vstack([(vec_i > vec_edges[:,0]) & (vec_i <= vec_edges[:,1]) for vec_i in vec]).sum(0)
    locs =  np.where(ar !=0)[0]
    if return_style == 'locs':
        return locs
    else:
       
        if return_style == 'array':            
            return ar
        
        elif return_style == 'all':
            return locs, ar, vec_edges
        
        else:
            raise ValueError('undefined return style')
        
        
        
        
def relative_eror(reco,real, return_mean = True, func = np.nanmean):
    """
    Calculate the relative reconstruction error
    Inputs:
        reco: k X T reconstructed dynamics matrix
        real: k X T real dynamics matrix (ground truth)
        return_mean: reaturn the average of the reconstruction error over time
        func: the function to apply on the relative error of each point
    Output:
        the relative error (or the mean relative error over time if return_mean)
    """
    error_point = np.sqrt(((reco - real)**2)/(real)**2)
    if return_mean:
        return func(error_point )
    return func(error_point,0)


def claculate_percent_close(reco, real, epsilon_close = 0.1, return_quantiles = False, quantiles = [0.05,0.95]):

    """
    Calculate the ratio of close (within a specific distance) points among all dynamics' points.
    
    Parameters:
    -----------
    reco: k x T numpy array
        The reconstructed dynamics matrix.
    real: k x T numpy array
        The real dynamics matrix (ground truth).
    epsilon_close: float, optional (default: 0.1)
        The threshold for distance.
    return_quantiles: bool, optional (default: False)
        Whether to return confidence interval values.
    quantiles: list of float, optional (default: [0.05, 0.95])
        The lower and higher limits for the quantiles.
    
    Returns:
    --------
    mean_close: float
        The mean of the close enough points.
    q1: float
        The first quantile (only returned if `return_quantiles` is True).
    q2: float
        The second quantile (only returned if `return_quantiles` is True).
    """
    close_enough = np.sqrt(np.sum((reco - real)**2,0)) < epsilon_close

    if return_quantiles:
        try:
            q1,q2 = stats.proportion.proportion_confint(np.sum(close_enough),len(close_enough),quantiles[0])
        except:
            q1 = np.mean(close_enough)
            q2 = np.mean(close_enough)
        return np.mean(close_enough), q1, q2
    return np.mean(close_enough)
    
# def load_mat_file(mat_name , mat_path = '',sep = sep):
#     """
#     Load a MATLAB `.mat` file. Useful for uploading the C. elegans data.
    
#     Parameters:
#     -----------
#     mat_name: str
#         The name of the MATLAB file.
#     mat_path: str, optional (default: '')
#         The path to the MATLAB file.
#     sep: str, optional (default: the system separator)
#         The separator to use in the file path.
    
#     Returns:
#     --------
#     data_dict: dict
#         A dictionary containing the contents of the MATLAB file.
#     """
    
#     data_dict = mat73.loadmat(mat_path+sep+mat_name)
#     return data_dict

import mat73
import scipy.io as sio
from scipy.sparse import coo_matrix  


import pandas as pd
import matplotlib.pyplot as plt

def save_df_as_fig(df, params_dict={}):
    fig_width = params_dict.get("figsize", (20, 5))[0]
    base_height = params_dict.get("row_height", 0.5)  # base row height
    df_str = df.astype(str)
    
    row_heights = [
        base_height + 0.02 * max(len(cell) for cell in row)
        for row in df_str.values
    ]
    total_height = sum(row_heights) + 1  # +1 for column headers

    fig, ax = plt.subplots(figsize=(fig_width, total_height))
    
    #fig, ax = plt.subplots(figsize=params_dict.get("figsize", (20, 5)))
    ax.axis('off')
    pd.plotting.table(
    ax,
    data=df.astype(str),  # <- pass as 'data'
    loc=params_dict.get("loc", "center"),
    colWidths=params_dict.get("colWidths", [0.2] * len(df.columns))
    )

    save_fig(
        params_dict.get("fig_name", "df_image"),
        fig,
        params_dict.get("save_path", os.getcwd()),
        save_params=params_dict.get("save_params", {'dpi': params_dict.get("dpi", 300)})
    )
    print('saved in %s' % params_dict.get("save_path", os.getcwd()))
    plt.close(fig)




def from_spike_times_to_rate(spike_dict, type_convert = 'discrete',
                             res = 0.01, max_min_val = [], return_T = False, 
                             T_max = np.inf, T_min = 0,  
                             params_gauss = {'wind' : 10, 'direction' : 1, 'sigma' : 1, 'norm_sum' : True, 'plot_gaussian' : False}):
    """
    Converts spike times to firing rates.
    spike dict is dictionary of units vs spike times
    res is how much to mutiply it by, such that each res bin will get 1 index. For instance, if res is 0.01 then each [0,1] will get 100 indices. 
    For instance, if my units are ms, an I want 20ms each index, I can have res = 20. 
    in this case it would be better to have res = 1, and then in params gauss have wind = 20 ms and sigma = 5 ms. 
    Parameters:
    - spike_dict (dict): A dictionary of units vs spike times.
    - res (float): A value by which to multiply the spike times.
    - type_convert (str): Type of conversion to perform (default is 'discrete').
    - Ts (dict): Dictionary containing time indices.
    - Ns (dict): Dictionary containing neuron indices.
    - firings_rates_gauss (dict): Dictionary containing Gaussian-convolved firing rates.
    - firings_rates (dict): Dictionary containing firing rates.
    - max_min_val (list): List containing minimum and maximum values.
    - return_T (bool): Whether to return firing rate matrices (default is False).
    - T_max (float): Maximum time value (default is np.inf).
    - params_gauss (dict): Dictionary containing parameters for Gaussian convolution.
    
    Returns:
    - firing_rate_mat (ndarray): Matrix containing firing rates.
    - firing_rate_mat_gauss (ndarray): Matrix containing Gaussian-convolved firing rates.
    - return_T (bool): Whether to return firing rate matrices.
    
    import numpy as np


    """  
    if isinstance(spike_dict , (np.ndarray, list)):
        spike_dict = {1: spike_dict}       
        
        
    if T_min >= T_max:
        raise ValueError('T_min must be larger than T_max')
    if res != 1:
        spike_dict = {key:np.array(val) / res for key,val in spike_dict.items()}
    if T_min > 0:
        spike_dict = {key:val - T_min for key,val in spike_dict.items()}
        spike_dict = {key : val[val > 0] for key,val in spike_dict.items()}
    print(type(spike_dict))
        
    """
    make sure keys are continues
    """
    if set(np.arange(len(spike_dict))) != set(list(spike_dict.keys())):
        new_keys = np.arange(len(spike_dict))
        old_keys = list(spike_dict.keys())
        old2new = {old:new for old,new in zip(old_keys, new_keys)}
        spike_dict = {old2new[key]:val for key,val in spike_dict.items()}
    else:
        old2new = {}
    
    
    
    if checkEmptyList(max_min_val):
        try:
            min_val = np.min([np.min(val) for val in list(spike_dict.values()) if len(val) > 0])
            max_val = np.max([np.max(val) for val in list(spike_dict.values()) if len(val) > 0])
        except:
            print(spike_dict)
        #min_max_val = [min_val, max_val]
        
        
    N = len(spike_dict)
    # if (min_val < 0 and T_min == 0) or T_min > 0:
    #     if T_min == 0:
    #         T_min = min_val
    #     spike_dict = {key : val - T_min for key,val in spike_dict.items()}
    #     spike_dict = {key : val[val > 0] for key,val in spike_dict.items()}
        
    if T_min > 0:
        max_val = max_val - T_min     
    max_val = int(np.ceil(max_val))
    max_val = int(np.min([max_val, T_max]))
    firing_rate_mat = np.zeros((int(N) ,max_val))    

        
    if type_convert == 'discrete':         
        T_thres = T_max #- T_min
        tup_neurons_and_spikes = np.vstack([ np.hstack([np.array([neuron]*np.sum( times < T_thres )).reshape((-1,1)) , np.array(times[ times < T_thres]).reshape((-1,1)) ])
                                  for neuron, times  in spike_dict.items()])
        rows =  tup_neurons_and_spikes[:,0]
        cols =  tup_neurons_and_spikes[:,1]
        
        data = np.ones(len(rows))  # Assuming all values are 1
        sparse_mat = coo_matrix((data, (rows, cols)), shape=(N, max_val))
        
        # for count, (neuron, times) in enumerate(spike_dict.items()):
                
        #     times_within = times.astype(int) #- max_min_per_file[neural_key][0]

        #     max_ind = times_within[(times_within > T_min ) & (times_within < T_max)]

        #     firing_rate_mat[count, max_ind] += 1

        firing_rate_mat = sparse_mat.toarray()
        firing_rate_mat_gauss = gaussian_convolve(firing_rate_mat,  **params_gauss)
            
    if T_min > 0 :     
        firing_rate_mat = firing_rate_mat[:, T_min:]
        firing_rate_mat_gauss = firing_rate_mat_gauss[:, T_min:]
    if return_T:
        return  firing_rate_mat, firing_rate_mat_gauss, return_T
    return  firing_rate_mat, firing_rate_mat_gauss, old2new

def split_to_trials(spikes_info = {}, firing_rate_mat = [], trial_start_array = [], trial_end_array = [], trial_word = 'trial', trial_key_type = 'str'):
    """
    Splits the provided spikes information and/or firing rate matrix into trials based on the given start and end times.
    
    Parameters:
    - spikes_info (dict, optional): Dictionary containing spike information for neurons. Default is an empty dictionary.
    - firing_rate_mat (list, optional): Matrix containing firing rates. Default is an empty list.
    - trial_start_array (array): Array containing the start times of the trials.
    - trial_end_array (array): Array containing the end times of the trials.
    - trial_word (str, optional): Prefix for trial keys if trial_key_type is 'str'. Default is 'trial'.
    - trial_key_type (str, optional): Type of keys for trials in the resulting dictionaries. Can be 'str' or 'int'. Default is 'str'.
    
    Returns:
    - trials_rate (dict): Dictionary containing the firing rate matrices for each trial.
    - trials_fire (dict): Dictionary containing the spike information for each trial.
    
    Raises:
    - ValueError: If no spikes or firing rate is provided, or if start and end times do not match, or if trial_key_type is not 'int' or 'str'.
    """
    
    # the function can split both dict and mat. if both provided - both. if one - only one.
    
    
    if len(spikes_info) == 0 and checkEmptyList(firing_rate_mat):
        raise ValueError('No spikes or firing rate provided!') 
        
    if len(trial_start_array) != len(trial_end_array):
        raise ValueError('start and end times must match!')
    if len(trial_end_array) == 0:
        raise ValueError('start or end cannot be empty!')
    
    ##########################################################
    if not checkEmptyList(firing_rate_mat):
        if  trial_key_type == 'str':
            trials_rate = {'%s_%d'%(trial_word, trial_count):firing_rate_mat[:, start:end]  for trial_count, (start, end) in enumerate(zip(trial_start_array, trial_end_array))}
        elif  trial_key_type == 'int':
            trials_rate = {trial_count : firing_rate_mat[:, start:end]  for trial_count, (start, end) in enumerate(zip(trial_start_array, trial_end_array))}
        else:
            raise ValueError('trial_key_type must be int or str')
    else:
        trials_rate = {}
        
        
    ##########################################################
    if not checkEmptyList(spikes_info):
        if  trial_key_type == 'str':
            trials_fire = {'%s_%d'%(trial_word, trial_count) : 
                           {neuron:times[(times <= end) & (times > start)] for neuron, times in spikes_info.items()}
                           for trial_count, (start, end) in enumerate(zip(trial_start_array, trial_end_array))}
            
            
        elif  trial_key_type == 'int':
            trials_fire = {trial_count: 
                           {neuron:times[(times <= end) & (times > start)] for neuron, times in spikes_info.items()}
                           for trial_count, (start, end) in enumerate(zip(trial_start_array, trial_end_array))}
            
        else:
            raise ValueError('trial_key_type must be int or str')
    else:
        trials_fire = {}
        
    return trials_rate, trials_fire
        


def merge_dicts(list_of_dicts):
    super_dict = {}
    for dict_i in list_of_dicts:
        super_dict = {**super_dict, **dict_i}
    return super_dict
            
def vstack_uneven(list_of_arrays):
    max_len = np.max([len(el) for el in list_of_arrays])
    num_ar = len(list_of_arrays)
    
    zers = np.zeros((num_ar, max_len))*np.nan
    for c, ar in enumerate(list_of_arrays):
        zers[c][:len(ar)] = ar
    return zers
    
def dstack_uneven(list_of_arrays, allow_unequal_rows = False, 
                  max_dur = 2000, fill_val = np.nan):
    max_len = np.max([el.shape[1] for el in list_of_arrays if len(el) > 0])
    max_len = np.min([max_len, max_dur])
    num_rows = np.array([el.shape[0] for el in list_of_arrays])
    num_rows[num_rows == 0] = np.max(num_rows)
    #if np.any([len(el) for el in list_of_arrays]) == 0:
            
        
    if np.any(num_rows[0] != num_rows):
        if allow_unequal_rows:
            max_rows = np.max(num_rows)
        else:
            raise ValueError('rows mismatch')
    else:
        max_rows = num_rows[0]
    
    num_ar = len(list_of_arrays)    
    zers = np.ones(( max_rows , max_len, num_ar))*fill_val
    
    for c, ar in enumerate(list_of_arrays):
        if len(ar) > 0:
            if ar.shape[1] > max_dur:
                ar = ar[:, :max_dur]
            zers[:ar.shape[0],:min([max_dur, ar.shape[1]]), c] = ar
    return zers
        


def create_unique_colors(cmap, n = 5 ):
    # Generate n evenly spaced values from 0 to 1
    colors = np.linspace(0, 1, n)

    # Get the 'hsv' colormap
    hsv = cm.get_cmap(cmap, n)

    # Map the color indices to the colormap
    color_list = hsv(colors)

    # Printing or returning the colors
    return color_list
    
    
    

        
        
# def gaussian_convolve(mat, wind = 10, direction = 1, sigma = 1):
#     """
#     Convolve a 2D matrix with a Gaussian kernel along the specified direction.
    
#     Parameters:
#         mat (numpy.ndarray): The 2D input matrix to be convolved with the Gaussian kernel.
#         wind (int, optional): The half-size of the Gaussian kernel window. Default is 10.
#         direction (int, optional): The direction of convolution. 
#             1 for horizontal (along columns), 0 for vertical (along rows). Default is 1.
#         sigma (float, optional): The standard deviation of the Gaussian kernel. Default is 1.
    
#     Returns:
#         numpy.ndarray: The convolved 2D matrix with the same shape as the input 'mat'.
        
#     Raises:
#         ValueError: If 'direction' is not 0 or 1.
#     """
#     if len(mat.shape) == 2:
#         if direction == 1:
#             gaussian = gaussian_array(2*wind,sigma)
#             mat_shape = mat.shape[1]
#             return np.vstack([ [np.sum(mat[row, np.max([t - wind,0]): np.min([t + wind, mat_shape])]*cut_gauss(gaussian, t, wind, left = 0, right = mat_shape)) 
#                          for t in range(mat.shape[1])] 
#                        for row in range(mat.shape[0])])
#         elif direction == 0:
#             return gaussian_convolve(mat.T, wind, direction = 1, sigma = sigma).T
#         else:
#             raise ValueError('invalid direction')
#     elif len(mat.shape) == 3 and direction >= 2:
#         raise ValueError('to be implemented convolve')
#     elif len(mat.shape) == 3:
#         return np.dstack([gaussian_convolve(mat[:,:,d], wind, direction, sigma ) 
#                           for d in range(mat.shape[2])])
#     else:
#         raise ValueError('?!')
            
        

def load_mat_file(mat_name , mat_path = '',sep = sep, squeeze_me = True,simplify_cells = True):
    """
    Function to load mat files. Useful for uploading the c. elegans data. 
    Example:
        load_mat_file('WT_Stim.mat','E:\CoDyS-Python-rep-\other_models')
    """
    try:
        if mat_path == '':
            data_dict = sio.loadmat(mat_name, squeeze_me = squeeze_me,simplify_cells = simplify_cells)
        else:
            data_dict = sio.loadmat(mat_path+sep+mat_name, squeeze_me = True,simplify_cells = simplify_cells)
    except: 
        try:
            data_dict = mat73.loadmat(mat_path+sep+mat_name)
        except:
            data_dict = scipy.io.loadmat(mat_path+sep+mat_name)
    return data_dict    
    


def min_dist(dotA1, dotA2, dotB1, dotB2, num_sects = 500):
    """
    Calculates the minimum euclidean distance between two discrete lines (e.g. where they intersect?).
    Inputs:
        dotA1: Tuple of x,y coordinate of first point on line A
        dotA2: Tuple of x,y coordinate of second point on line A
        dotB1: Tuple of x,y coordinate of first point on line B
        dotB2: Tuple of x,y coordinate of second point on line B
        num_sects: Number of sections the lines should be divided into to calculate distance
        
    Returns:
        List of minimum distances between two lines.
    """    
    x_lin = np.linspace(dotA1[0], dotA2[0])
    y_lin = np.linspace(dotA1[1], dotA2[1])
    x_lin_or = np.linspace(dotB1[0], dotB2[0])
    y_lin_or = np.linspace(dotB1[1], dotB2[1])
    dist_list = []
    for pairA_num, pairAx in enumerate(x_lin):
        pairAy = y_lin[pairA_num]
        for pairB_num, pairBx in enumerate(x_lin_or):
            pairBy = y_lin_or[pairB_num]
            dist = (pairAx - pairBx)**2 + (pairAy - pairBy)**2
            dist_list.append(dist)
    return dist_list         
#%% FHN model
# taken from https://www.normalesup.org/~doulcier/teaching/modeling/excitable_systems.html    
    
def create_FHN(dt = 0.01, max_t = 100, I_ext = 0.5,
               b = 0.7, a = 0.8 , tau = 20, v0 = -0.5, w0 = 0, params = {'exp_power' : 0.9, 'change_speed': False}):
    
    """
    Create the FitzHugh-Nagumo dynamics
    Inputs:
        dt: time step
        max_t: maximum time
        I_ext: external current
        b: model parameter
        a: model parameter
        tau: model parameter
        v0: initial condition for v
        w0: initial condition for w
        params: dictionary of additional parameters
            exp_power: power to raise time to for non-uniform time
            change_speed: Boolean to determine whether to change time speed
    Returns:
        v_full: list of v values at each time step
        w_full: list of w values at each time step
    """
    
    time_points = np.arange(0, max_t, dt)
    if params['change_speed']:
        time_points = time_points**params['exp_power']
    
        
    w_full = []
    v_full = []
    v = v0
    w = w0
    for t in time_points:
        v, w =  cal_next_FHN(v,w, dt , max_t , I_ext , b, a , tau)
        v_full.append(v)
        w_full.append(w)
    return v_full, w_full


        
def cal_next_FHN(v,w, dt = 0.01, max_t = 300, I_ext = 0.5, 
                 b = 0.7, a = 0.8 , tau = 20) :
    """
    Calculate next v and w values for FitzHugh-Nagumo dynamics
    Inputs:
        v: current v value
        w: current w value
        dt: time step
        max_t: maximum time
        I_ext: external current
        b: model parameter
        a: model parameter
        tau: model parameter
    Returns:
        v_next: next v value
        w_next: next w value
    """    
    v_next = v + dt*(v - (v**3)/3 - w + I_ext)
    w_next = w + dt/tau*(v + a - b*w)
    return v_next, w_next 

def norm_over_time(coefficients, type_norm = 'normal'):
    """
    Normalize coefficients over time
    Inputs:
        coefficients: array of coefficients
        type_norm: type of normalization
            'normal': standard normalization
    Returns:
        coefficients_norm: normalized coefficients
    """    
    if type_norm == 'normal':
        coefficients_norm = (coefficients - np.mean(coefficients,1).reshape((-1,1)))/np.std(coefficients, 1).reshape((-1,1))
    return coefficients_norm

def normalize_data(data, style_normalize='minmax', axis=1):
    data = np.asarray(data)
    original_shape = data.shape

    if data.ndim == 1:
        data = data.reshape(-1, 1)
        axis = 0

    if style_normalize == 'minmax':
        min_val = np.nanmin(data, axis=axis, keepdims=True)
        max_val = np.nanmax(data, axis=axis, keepdims=True)
        norm = (data - min_val) / (max_val - min_val + 1e-8)

    elif style_normalize == 'zscore':
        mean = np.nanmean(data, axis=axis, keepdims=True)
        std = np.nanstd(data, axis=axis, keepdims=True)
        norm = (data - mean) / (std + 1e-8)

    elif style_normalize == 'robust':
        median = np.nanmedian(data, axis=axis, keepdims=True)
        q1 = np.nanpercentile(data, 25, axis=axis, keepdims=True)
        q3 = np.nanpercentile(data, 75, axis=axis, keepdims=True)
        iqr = q3 - q1 + 1e-8
        norm = (data - median) / iqr

    elif style_normalize == 'maxabs':
        max_abs = np.nanmax(np.abs(data), axis=axis, keepdims=True)
        norm = data / (max_abs + 1e-8)

    elif style_normalize == 'robustmaxabs':
        max_abs = np.nanpercentile(np.abs(data), 99, axis=axis, keepdims=True)
        norm = data / (max_abs + 1e-8)

    else:
        raise ValueError(f"Unsupported normalization style: {style_normalize}")

    return norm.reshape(original_shape)


def find_perpendicular(d1, d2, perp_length = 1, prev_v = [], next_v = [], ref_point = [],choose_meth = 'intersection',initial_point = 'mid',  
                       direction_initial = 'low', return_unchose = False, layer_num = 0):
    """
    IT IS AN INTER FUNCTION - DO NOT USE IT BY ITSELF
    This function find the 2 point of the orthogonal vector to a vector defined by points d1,d2
    d1 =                first data point
    d2 =                second data point
    perp_length =       desired width
    prev_v =            previous value of v. Needed only if choose_meth == 'prev'
    next_v =            next value of v. Needed only if choose_meth == 'prev'
    ref_point =         reference point for the 'smooth' case, or for 2nd+ layers
    choose_meth =       'intersection' (eliminate intersections) OR 'smooth' (smoothing with previous prediction) OR 'prev' (eliminate convexity)
    direction_initial = to which direction take the first perp point  
    return_unchose =    whether to return unchosen directions   
    
    """       
    # Check Input    
    if d2[0] == d1[0] and d2[1] == d1[1]:
        raise ValueError('d1 and d2 are the same point')
    
    # Define start point for un-perp curve
    if initial_point == 'mid':
        perp_begin = (np.array(d1) + np.array(d2))/2
        d1_perp = perp_begin
    elif initial_point == 'end':        d1_perp = d2
    elif initial_point == 'start':        d1_perp = d1
    else:        raise NameError('Unknown intial point')       
    
    # If perpendicular direction is according to 'intersection' elimination
    if choose_meth == 'intersection':
        if len(prev_v) > 0:        intersected_curve1 = prev_v
        else:                      intersected_curve1 = d1
        if len(next_v) > 0:        intersected_curve2 = next_v
        else:                      intersected_curve2 = d2
        
    # If a horizontal line       
    if d2[0] == d1[0]:        d2_perp = np.array([d1_perp[0]+perp_length, d1_perp[1]])
    # If a vertical line
    elif d2[1] == d1[1]:        d2_perp = np.array([d1_perp[0], d1_perp[1]+perp_length])
    else:
        m = (d2[1]-d1[1])/(d2[0]-d1[0]) 
        m_per = -1/m                                       # Slope of perp curve        
        theta1 = np.arctan(m_per)
        theta2 = theta1 + np.pi
        
        # if smoothing
        if choose_meth == 'smooth' or choose_meth == 'intersection':
            if len(ref_point) == 0: 
                #print('no ref point!')
                smooth_val =[]
            else:                smooth_val = np.array(ref_point)
        
        # if by convexity
        if choose_meth == 'prev':
            if len(prev_v) > 0 and len(next_v) > 0:                     # both sides are provided
                prev_mid_or = (np.array(prev_v) + np.array(next_v))/2
            elif len(prev_v) > 0 and len(next_v) == 0:                  # only the previous side is provided
                prev_mid_or = (np.array(prev_v) + np.array(d2))/2
            elif len(next_v) > 0 and len(prev_v) == 0:                  # only the next side is provided               
                prev_mid_or = (np.array(d1) + np.array(next_v))/2
            else:
                raise ValueError('prev or next should be defined (to detect convexity)!')        

        if choose_meth == 'prev':
            prev_mid = prev_mid_or
        elif choose_meth == 'smooth':
            prev_mid = smooth_val
        elif choose_meth == 'intersection':
            prev_mid = smooth_val
            
        x_shift = perp_length * np.cos(theta1)
        y_shift = perp_length * np.sin(theta1)
        d2_perp1 = np.array([d1_perp[0] + x_shift, d1_perp[1]+ y_shift])            
        
        x_shift2 = perp_length * np.cos(theta2)
        y_shift2 = perp_length * np.sin(theta2)
        d2_perp2 = np.array([d1_perp[0] + x_shift2, d1_perp[1]+ y_shift2])
        options_last = [d2_perp1, d2_perp2]        
        # Choose the option that goes outside
        if len(prev_mid) > 0:            
          
            if len(ref_point) > 0 and layer_num > 0:                               # here ref point is a point of a different dynamics layer from which we want to take distance
                dist1 = np.sum((smooth_val - d2_perp1)**2)
                dist2 = np.sum((smooth_val - d2_perp2)**2)
                max_opt = np.argmax([dist1, dist2])

            elif choose_meth == 'intersection':
                dist1 = np.min(min_dist(prev_mid, d2_perp1, intersected_curve1, intersected_curve2))
                dist2 = np.min(min_dist(prev_mid, d2_perp2, intersected_curve1, intersected_curve2))
                max_opt = np.argmax([dist1,dist2]) 
         
            else:
                dist1 = np.sum((prev_mid - d2_perp1)**2)
                dist2 = np.sum((prev_mid - d2_perp2)**2)
                max_opt = np.argmin([dist1,dist2])       
                
        else:
        
            if len(ref_point) > 0 and layer_num >0:                               # here ref point is a point of a different dynamics layer from which we want to take distance
                dist1 = np.sum((ref_point - d2_perp1)**2)
                dist2 = np.sum((ref_point - d2_perp2)**2)
                max_opt = np.argmax([dist1, dist2])             
            elif direction_initial == 'low':
                max_opt = np.argmin([d2_perp1[1], d2_perp2[1]])
            elif direction_initial == 'high':
                max_opt = np.argmax([d2_perp1[1], d2_perp2[1]])
            elif direction_initial == 'right' :
                max_opt = np.argmax([d2_perp1[0], d2_perp2[0]])
            elif direction_initial == 'left':
                max_opt = np.argmin([d2_perp1[0], d2_perp2[0]])                
            else:
                raise NameError('Invalid direction initial value') 
    
    d2_perp = options_last[max_opt] # take the desired direction
    if return_unchose:
        d2_perp_unchose = options_last[np.abs(1 - max_opt)] 
        return d1_perp, d2_perp, d2_perp_unchose
    return d1_perp, d2_perp


def find_lows_high(coeff_row, latent_dyn,   choose_meth ='intersection',factor_power = 0.9, initial_point = 'start',
                   direction_initial = 'low', return_unchose = False, ref_point = [], layer_num = 0):
    """
    IT IS AN INTER FUNCTION - DO NOT USE IT BY ITSELF
    Calculates the coordinates of the 'high' values of a specific kayer
    """
    
    if return_unchose: unchosen_highs = []
    ### Initialize
    x_highs_y_highs = []; x_lows_y_lows = []
    if isinstance(ref_point, np.ndarray):
        if len(ref_point.shape) > 1:
            ref_shape_all = ref_point
        else:
            ref_shape_all = np.array([])
    else:
        ref_shape_all = np.array([])
    # Iterate over time
    for t_num in range(0,latent_dyn.shape[1]-2): 
  
        d1_coeff = latent_dyn[:,t_num]
        d2_coeff = latent_dyn[:,t_num+1]
        prev_v = latent_dyn[:,t_num-1] 
        next_v = latent_dyn[:,t_num+2]
        c_len = (coeff_row[t_num] + coeff_row[t_num+1])/2

        if len(ref_shape_all) > 0 and ref_shape_all.shape[0] > t_num and layer_num > 0: # and ref_shape_all.shape[1] >1
            ref_point = ref_shape_all[t_num,:]
          
            if len(ref_point) >  0 and layer_num > 0 :  #and t_num  < 3
                 pass        
        
        # if do not consider layer        
        elif t_num > 2 and (choose_meth == 'smooth' or choose_meth == 'intersection'):   
            ref_point  = d2_perp         
        else:              
            ref_point = []          
        if return_unchose:  d1_perp, d2_perp, d2_perp_unchosen = find_perpendicular(d1_coeff, d2_coeff,c_len**factor_power, prev_v = prev_v, next_v=next_v,ref_point  = ref_point , choose_meth = choose_meth, initial_point=initial_point, direction_initial =direction_initial, return_unchose = return_unchose,layer_num=layer_num)# c_len
        else:               d1_perp, d2_perp = find_perpendicular(d1_coeff, d2_coeff,c_len**factor_power, prev_v = prev_v, next_v=next_v,ref_point  = ref_point , choose_meth = choose_meth, initial_point=initial_point, direction_initial= direction_initial, return_unchose = return_unchose,layer_num=layer_num)# c_len
        # Add results to results lists
        x_lows_y_lows.append([d1_perp[0],d1_perp[1]])
        x_highs_y_highs.append([d2_perp[0],d2_perp[1]])
        if return_unchose: unchosen_highs.append([d2_perp_unchosen[0],d2_perp_unchosen[1]])
    # return
    if return_unchose: 
        return x_lows_y_lows, x_highs_y_highs, unchosen_highs
    return x_lows_y_lows, x_highs_y_highs        


def plot_multi_colors(store_dict,min_time_plot = 0,max_time_plot = -100,  colors = ['green','red','blue'], ax = [],
                      fig = [], alpha = 0.99, smooth_window = 3, factor_power = 0.9, coefficients_n = [], to_scatter = False, 
                      to_scatter_only_one = False ,choose_meth = 'intersection', title = ''):
    """
    store_dict is a dictionary with the high estimation results. 
    example:        
        store_dict , coefficients_n = calculate_high_for_all(coefficients,choose_meth = 'intersection',width_des = width_des, latent_dyn = latent_dyn, direction_initial = direction_initial,factor_power = factor_power, return_unchose=True)
    
    """
    if len(colors) < len(store_dict):                raise ValueError('Not enough colors were provided')
    if isinstance(ax, list) and len(ax) == 0:        fig, ax = plt.subplots(figsize = (20,20))
    for key_counter, (key,set_plot) in enumerate(store_dict.items()):
        if key_counter == 0:
            x_lows_y_lows = store_dict[key][0]
            x_highs_y_highs = store_dict[key][1]
            #choose_meth_initial = 
            low_ref =[]
            high_ref = []
        else:
            low_ref = [np.array(x_highs_y_highs)[min_time_plot,0],   np.array(x_highs_y_highs)[min_time_plot,1]]
            high_ref = [np.array(x_highs_y_highs)[max_time_plot,0],np.array(x_highs_y_highs)[max_time_plot,1]]
        if len(coefficients_n) > 0:
            # Define the length of the last perp. 
            c_len = (coefficients_n[key,max_time_plot-1] + coefficients_n[key,max_time_plot])/2
            # Create perp. in the last point            
            d1_p, d2_p =find_perpendicular([np.array(x_lows_y_lows)[max_time_plot-2,0],np.array(x_lows_y_lows)[max_time_plot-2,1]], 
                                           [np.array(x_lows_y_lows)[max_time_plot-1,0],np.array(x_lows_y_lows)[max_time_plot-1,1]], 
                                           perp_length = c_len**factor_power, 
                                           ref_point = high_ref, 
                                           choose_meth = 'intersection',initial_point = 'end')
            # Define the length of the first perp. 
            c_len_start = (coefficients_n[key,min_time_plot-1] + coefficients_n[key,min_time_plot])/2
            # Create perp. in the first point   
            d1_p_start =[np.array(x_highs_y_highs)[min_time_plot,0],np.array(x_highs_y_highs)[min_time_plot,1]]
                                                       
            d2_p_start=  [np.array(x_highs_y_highs)[min_time_plot+1,0],np.array(x_highs_y_highs)[min_time_plot+1,1]]                                                        

            x_lows_y_lows = store_dict[key][0]
            x_highs_y_highs = store_dict[key][1] 

            stack_x = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,0].flatten(), np.array([d2_p[0]]), np.array(x_highs_y_highs)[max_time_plot-1:min_time_plot+1:-1,0].flatten(),np.array([d2_p_start[0]])])
            stack_y = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,1].flatten(), np.array([d2_p[1]]),np.array(x_highs_y_highs)[max_time_plot-1:min_time_plot+1:-1,1].flatten(),np.array([d2_p_start[1]])])
            
        else:
            stack_x = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,0].flatten(), np.array(x_highs_y_highs)[max_time_plot:min_time_plot:,0].flatten()])
            stack_y = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,1].flatten(), np.array(x_highs_y_highs)[max_time_plot:min_time_plot:,1].flatten()])
        stack_x_smooth = stack_x
        stack_y_smooth = stack_y
        if key_counter !=0:
            ax.fill(stack_x_smooth, stack_y_smooth, alpha=0.3, facecolor=colors[key_counter], edgecolor=None, zorder=1, snap = True)#
        else:
            ax.fill(stack_x_smooth, stack_y_smooth, alpha=alpha, facecolor=colors[key_counter], edgecolor=None, zorder=1, snap = True)#

    if to_scatter or (to_scatter_only_one and key == np.max(list(store_dict.keys()))):
        

          ax.scatter(np.array(x_lows_y_lows)[min_time_plot:max_time_plot,0].flatten(), np.array(x_lows_y_lows)[min_time_plot:max_time_plot,1].flatten(), c = 'black', alpha = alpha, s = 45)

    remove_edges(ax)
    if not title  == '':
        ax.set_title(title, fontsize = 20)
    


def remove_edges(ax, include_ticks = True, top = False, right = False, bottom = True, left = True):
    """
    Remove the specified edges (spines) of the plot and optionally the ticks of the plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object of the plot.
    include_ticks : bool, optional
        Whether to include the ticks, by default False.
    top : bool, optional
        Whether to remove the top edge, by default False.
    right : bool, optional
        Whether to remove the right edge, by default False.
    bottom : bool, optional
        Whether to remove the bottom edge, by default False.
    left : bool, optional
        Whether to remove the left edge, by default False.
    
    Returns
    -------
    None
    """    
    ax.spines['top'].set_visible(top)    
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)  
    if not include_ticks:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

def norm_coeffs(coefficients, type_norm, 
                same_width = True,width_des = 0.7,
                factor_power = 0.9, min_width = 0.01):
    """
    Normalize the coefficients according to the specified type of normalization.
    
    Parameters
    ----------
    coefficients : numpy.ndarray
        The coefficients to be normalized.
    type_norm : str
        The type of normalization to be applied. Can be 'sum_abs', 'norm', 'abs' or 'no_norm'.
    same_width : bool, optional
        Whether to enforce the same width for all coefficients, by default True.
    width_des : float, optional
        The desired width, by default 0.7.
    factor_power : float, optional
        The power factor to apply, by default 0.9.
    min_width : float, optional
        The minimum width allowed, by default 0.01.
    
    Returns
    -------
    numpy.ndarray
        The normalized coefficients.
    
    Raises
    ------
    NameError
        If the `type_norm` value is not one of the allowed values ('sum_abs', 'norm', 'abs' or 'no_norm').
    """
    if type_norm == 'norm':
        coefficients_n =      norm_over_time(np.abs(coefficients), type_norm = 'normal')   
        coefficients_n =      coefficients_n - np.min(coefficients_n,1).reshape((-1,1))
        #plt.plot(coefficients_n.T)
    elif type_norm == 'sum_abs':
        coefficients[np.abs(coefficients) < min_width] = min_width
        coefficients_n = np.abs(coefficients) / np.sum(np.abs(coefficients),1).reshape((-1,1))
    elif type_norm == 'abs':
        coefficients[np.abs(coefficients) < min_width] = min_width
        coefficients_n = np.abs(coefficients) 
    elif type_norm == 'no_norm':
        coefficients_n = coefficients
    else:
        raise NameError('Invalid type_norm value')


    coefficients_n[coefficients_n < min_width]  = min_width
    if same_width:        coefficients_n = width_des*(np.abs(coefficients_n)**factor_power) / np.sum(np.abs(coefficients_n)**factor_power,axis = 0)   
    else:                 coefficients_n = np.abs(coefficients_n) / np.sum(np.abs(coefficients_n),axis = 0)  
    coefficients_n[coefficients_n < min_width]  = min_width
    return coefficients_n


def plot_quiver_for_transition(A, min_grid = -2, max_grid = 2,
                               num_points = 10, ax = [], plot_params = {},
                               scale = 50, to_add_labels = True):
    # Create a grid of points
    x_values = np.linspace(min_grid, max_grid, num_points)
    y_values = np.linspace(min_grid, max_grid, num_points)
    X, Y = np.meshgrid(x_values, y_values)

    # Compute the transformation
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y

    # Normalize the vectors
    magnitude = np.sqrt(U**2 + V**2)
    U = U / (magnitude + 1e-6)
    V = V / (magnitude + 1e-6)
    
    
    # Plot the quiver
    if checkEmptyList(ax):
        fig, ax = plt.subplots()
        
    ax.quiver(X, Y, U, V, scale = scale, **plot_params)#, headwidth=5, headlength=7, headaxislength=6, minlength=0.1,*plot_params)
    
    if to_add_labels:
        ax.set_xlabel('$\delta x$')
        ax.set_ylabel('$\delta y$')
        
        
        
def lists2list(xss):
    """
    Flatten a list of lists into a single list.
    
    Parameters
    ----------
    xss : list of lists
        The list of lists to be flattened.
    
    Returns
    -------
    list
        The flattened list.
    """
    return [x for xs in xss for x in xs] 

def mean_change(signal, axis = 0):
    """
    Calculate the mean change of the signal along the specified axis.
    
    Parameters
    ----------
    signal : numpy.ndarray
        The signal data.
    axis : int, optional
        The axis along which the mean change is calculated, by default 0.
    
    Returns
    -------
    numpy.ndarray
        The mean change of the signal.
    """     
    return np.mean(np.abs(np.diff(signal, axis = axis)), axis = axis)
    



def kmeans_with_cross_labels():
    # this finds clusters for each groups with some clusters shared among groups
    pass
    
