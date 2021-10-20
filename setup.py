# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:51:33 2021

@author: ktopo
"""
from setuptools import setup

setup(
    name='codebase',
    version='0.1.0',    
    description='Python Package for Carbon Atmospheric Prediction',
    url='https://github.com/ktopolov/ECE537',
    author='Kenny Topolovec',
    author_email='ktopolov@umich.edu',
    license='BSD 2-clause',
    packages=['codebase'],
    install_requires=['mpi4py>=2.0',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
