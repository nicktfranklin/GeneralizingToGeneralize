# Simulations and analyses for [PAPER NAME]


#### Notebooks:



___


### Installation Instructions

This library run on Python 2.7 and unlike most python code, requries
 compilation with Cython before use. This requires a C compiler (gcc), 
 [for which you can find documentation here.](
 http://cython.readthedocs.io/en/latest/src/quickstart/install.html)  

 The needed libraries are listed in the file `enviornments.yml`. If you have conda installed,
  this file can be used to create a virtual enviornment cython and the other dependencies:  
 ```conda env create --file environment.yml```
 
 This will create an envionment named "compgen", which you can activate via:
 ```source activate compgen```

 The cython code will need be compiled manually:  
 ```python setup.py build_ext --inplace```  
  
### Files:
---
