# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 04:06:34 2023

@author: noga mudrik
"""


"""
Imports
"""
import matplotlib
from webcolors import name_to_rgb
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

sns.set_context('talk')
"""
parameters
"""
labelpad = 10

def str_dict2dict(string):
  string_val = string.replace('"','')
  sub_vals = string_val.replace('{','').replace('}','').split(',')
  sub_sub_vals = [val.split(':') for val in sub_vals]
  #print(sub_sub_vals)
  if np.array([len(el) == 2 for el in sub_sub_vals]).all():
    dict_return  = {val[0].strip():val[1].strip() for val in sub_sub_vals}
    return dict_return
  else:
    return {}



def create_3d_ax(num_rows, num_cols, params = {}):
    fig, ax = plt.subplots(num_rows, num_cols, subplot_kw = {'projection': '3d'}, **params)
    return  fig, ax


def plot_3d(mat, params_fig = {}, fig = [], ax = [], params_plot = {}, type_plot = 'plot'):
    #
    if checkEmptyList(ax):
        fig, ax = create_3d_ax(1,1, params_fig)
    if type_plot == 'plot':
        ax.plot(mat[0], mat[1], mat[2], **params_plot)
    else:
        ax.scatter(mat[0], mat[1], mat[2], **params_plot)

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

def create_colors(len_colors, perm = [0,1,2]):
    """
    Create a set of discrete colors with a one-directional order
    Input: 
        len_colors = number of different colors needed
    Output:
        3 X len_colors matrix decpiting the colors in the cols
    """
    colors = np.vstack([np.linspace(0,1,len_colors),(1-np.linspace(0,1,len_colors))**2,1-np.linspace(0,1,len_colors)])
    colors = colors[perm, :]
    return colors


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
    
def load_mat_file(mat_name , mat_path = '',sep = sep):
    """
    Load a MATLAB `.mat` file. Useful for uploading the C. elegans data.
    
    Parameters:
    -----------
    mat_name: str
        The name of the MATLAB file.
    mat_path: str, optional (default: '')
        The path to the MATLAB file.
    sep: str, optional (default: the system separator)
        The separator to use in the file path.
    
    Returns:
    --------
    data_dict: dict
        A dictionary containing the contents of the MATLAB file.
    """
    
    data_dict = mat73.loadmat(mat_path+sep+mat_name)
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
    


def remove_edges(ax, include_ticks = False, top = False, right = False, bottom = False, left = False):
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

def norm_coeffs(coefficients, type_norm, same_width = True,width_des = 0.7,factor_power = 0.9, min_width = 0.01):
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
    
