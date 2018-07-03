# Simulations and analyses for [PAPER NAME]

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
  
## Files List
* For each experiment in the paper, there are two jupyter notebooks, one for simulations 
(Generative Modeling) and another for human subject behavior (Analysis). For clarity, each is labeled
by the number of goals in the experimental design.

* `Exclusion Criteria.ipynb` contains the analyses used to exclude individual subjects. There
is one notebook for all three 

* The folder `models` contains the code for the computational models (e.g. joint clustering, 
independent clustering, etc.), as well as the code need to generate experiments for simulations

* `opt` contains various analysis functions used in the analyesis. This is done to maintain
 readability in the jupyter notebooks
 
* `data` contains deidentified subject data.

* `psiTurk` contains the code to run the behavioral experiment. These are coded in javascript and
 run on Amazon's mechanical turk via the psiTurk library [https://psiturk.org]. 


