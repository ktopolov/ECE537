# -*- coding: utf-8 -*-
"""
Model utility tools
"""
import numpy as np
from tensorflow import keras
import pickle

from codebase import features

# %% Prediction Wrappers
# Base Class
class WrapperModel():
    """Wrapper model class. Currently supports wrapping for TensorFlow and
    sklearn models
    """
    def __init__(self):
        """Initialize"""
        self.Model = None
        self.model_type = None

    def init_from_model(self, Model):
        """Initialize wrapper from a Model either from sklearn or TensorFlow
        
        Parameters
        ----------
        Model :
            Fitted or unfitted model
        """
        self.Model = Model

        model_type_full = str(type(Model))
        if 'sklearn' in model_type_full:
            self.model_type = 'sklearn'
        elif 'tensorflow' in model_type_full:
            self.model_type = 'tf'
        else:
            raise ValueError('Unknown Model type {}'.format(model_type_full))

    def init_from_file(self, model_type, file):
        """Initialize wrapper from a Model either from sklearn or TensorFlow
        
        Parameters
        ----------
        model_type : str
            Either 'sklearn' or 'tf'

        file : str
            Filepath
        """
        if model_type == 'sklearn':
            self.model_type = 'sklearn'
            self.Model = pickle.load((open(file, 'rb')))
        elif model_type == 'tf':
            self.model_type = 'tf'
            self.Model = keras.models.load_model(file)
        else:
            raise ValueError('model_type {} unknown'.format(model_type))

    def save(self, file):
        """Save model to file

        Parameters
        ----------
        file : str
            Filepath
        """
        if self.model_type == 'sklearn':
            pickle.dump(self.Model, open(file, 'wb'))
        elif self.model_type == 'tf':
            self.Model.save(filepath=file)
        else:
            raise ValueError('self.model_type {} either unset or invalid'.format(
                self.model_type))

    def predict(self, x):
        """Predict carbon for combination of given latitude, longitude, months

        Parameters
        ----------
        x : (..., M) np.ndarray float
            Input with arbitrary number of front dimensions and M features

        Returns
        -------
        y : (N) np.ndarray float
            Predicted output
        """
        if np.ndim(x) == 1:
            y = self.Model.predict(x[np.newaxis, :])
            y = y.flatten()
        elif np.ndim(x) == 2:
            y = self.Model.predict(x)
        elif np.ndim(x) > 2:
            orig_shape = x.shape
            new_shape = (np.prod(orig_shape[:-1]), orig_shape[-1])
            y = self.Model.predict(x.reshape(new_shape))
            y = y.reshape(orig_shape[:-1])
        else:
            raise ValueError('What are you doing... dimensions of x not right')

        return y

    def fit(self, x, y, **kwargs):
        """Fit model

        Parameters
        ----------
        x : (N, M) np.ndarray float
            N rows of M-feature input data

        y : (N) np.ndarray float
            N rows of output data

        kwargs : {}
            Additional model kwargs based on self.model_type
        """
        self.Model.fit(x, y)

    # def predict_grid(self, x_grid):
    #     """Predict carbon for combination of given latitude, longitude, months
    
    #     Parameters
    #     ----------
    #     Model :
    #         Any model type with a "y = predict(X=x)" function
            
    #     lats : (n_lat) np.ndarray float
    #         Latitudes
    
    #     lons : (n_lon) np.ndarray float
    #         Longitudes
    
    #     months : (n_month) np.ndarray float
    #         Months since 2003
    
    #     Returns
    #     -------
    #     carbon_predict : (n_lat, n_lon, n_month) np.ndarray float
    #         Predicted carbon
    #     """
    #     # Define grid of data
    #     lats_grid, lons_grid, months_grid = np.meshgrid(
    #         lats,
    #         lons,
    #         months,
    #         indexing='ij'
    #     )
    #     grid_shape = months_grid.shape
        
    #     months_grid = months_grid.flatten()
    #     lats_grid = lats_grid.flatten()
    #     lons_grid = lons_grid.flatten()
    
    #     # Transform to features
    #     ecf_grid = features.lla_to_ecf(lat=lats_grid, lon=lons_grid)
    #     norm_ecf_grid = ecf_grid / features.EARTH_RADIUS
        
    #     X = np.concatenate(
    #         (norm_ecf_grid, months_grid[:, np.newaxis]),
    #         axis=-1
    #     )

    #     carbon_predict = self.Model.predict(X)
    #     carbon_predict = carbon_predict.reshape(grid_shape)
    #     return carbon_predict

    
