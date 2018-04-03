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
  
## Todo!
* Match the experiment number to match the manuscript draft. As they are labeled here,  
    * Experiment 1: 3 Goals, 9 contexts
    * Experiment 2: 4 Goals, 10 Contexts
    * Experiment 3: 2 Goals, 6 contexts
    
* The analyses for experiment 3 will be updated as I fill in that section
* Need to write a description with the list of files
---
