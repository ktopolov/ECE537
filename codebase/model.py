# -*- coding: utf-8 -*-
"""
Model utility tools
"""
import numpy as np
from tensorflow import keras
import pickle

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
        self.__ready = False  # ready once initialized from model

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

        self.__ready = True

    def init_from_file(self, model_type, path):
        """Initialize wrapper from a Model either from sklearn or TensorFlow
        
        Parameters
        ----------
        model_type : str
            Either 'sklearn' or 'tf'

        path : str
            Filepath for sklearn models or directory for TF models
        """
        if model_type == 'sklearn':
            self.model_type = 'sklearn'
            self.Model = pickle.load((open(path, 'rb')))
        elif model_type == 'tf':
            self.model_type = 'tf'
            self.Model = keras.models.load_model(path)
        else:
            raise ValueError('model_type {} unknown'.format(model_type))

        self.__ready = True

    def ready(self):
        """Whether model is ready for prediction

        Returns
        -------
        ready : bool
        """
        return self.__ready

    def save(self, path):
        """Save model to file

        Parameters
        ----------
        path : str
            Filepath for sklearn models or directory for tf models
        """
        if self.model_type == 'sklearn':
            pickle.dump(self.Model, open(path, 'wb'))
        elif self.model_type == 'tf':
            keras.models.save_model(
                model=self.Model,
                filepath=path,
                overwrite=True,
            )
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
        y : (...) np.ndarray float
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

        Returns
        -------
        out :
            Output from self.Model.fit()
        """
        out = self.Model.fit(x, y, **kwargs)
        return out
