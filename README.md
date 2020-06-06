# loopfit
This package implements a fitting routine for superconducting resonator IQ loops. It is written in C++ with Ceres Solver
to prioritize speed.

## Getting Started
These instructions will help you install _loopfit_ on your machine. The code is still in it's beta stage so there is 
currently no release version. It must be downloaded and installed directly from GitHub.

### Prerequisites
#### System
No testing is currently implemented to verify cross platform compatibility, but the code is expected to be platform 
independent. Development was done on Mac OSX and Scientific Linux.  

#### Python
The code is designed to run on all python versions greater than 3.7. All other python prerequisites will be 
automatically downloaded during the installation. 

#### Ceres Solver
The Ceres Solver C++ fitting library must be installed prior to installation of this package. 

##### In a Conda Environment:
Run ```conda install -c conda-forge ceres-solver cxx-compiler```. The 
precompiled fitting code will be downloaded along with the appropriate C++ compiler with which to build the 
```loopfit``` module. The compiler will become the default for the environment and no more work is necessary.

##### Custom Installation:
Ceres Solver can also be compiled from the source files and installed to a system location. This allows ```loopfit``` to
function outside of a conda environment or if a custom build of Ceres Solver is required. See 
[these instructions](http://ceres-solver.org/installation.html) for more details.

To avoid linker errors, the same c and c++ compilers used to compile Ceres-Solver must also be used to compile 
```loopfit```. Set the $CXX and $CC environment variables for your system to ensure the appropriate compilers are used.

#### Git [optional]
Git must be installed to clone the repository (as in the install instructions), but it can be downloaded directly from 
[GitHub](https://github.com/zobristnicholas/loopfit) as well.


### Installing
On the command line run the following with your choice of \<directory\> and \<version\>:
```
cd <directory>
git clone --branch <version> https://github.com/zobristnicholas/loopfit.git
pip install loopfit
```
- In the first line choose the directory where you want the code to exist.
- In the second line choose the [version](https://github.com/zobristnicholas/loopfit/tags) that you want to install. 
(e.g. 0.1)
- The third line will install the code. Checking out other versions with git (```git checkout <version>```) requires 
```pip install loopfit``` to be rerun since the library must be recompiled.

## Versions
The versions of this code follow the [PEP440](https://www.python.org/dev/peps/pep-0440/) specifications.

Each version is given a git [tag](https://github.com/zobristnicholas/loopfit/tags) corresponding to a particular 
commit. Only these commits should be used since the others may not be stable. Versions prior to 1.0 should be considered
in the beta stage and subject to changes in the code's API.

## License 
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements 
This code is build on top of and owes its performance to the wonderful C++ fitting library 
[Ceres Solver](http://ceres-solver.org).

The initial guess for the loop fit was developed starting from the method used in the
[scraps package](https://github.com/FaustinCarter/scraps). 

The code implementing the [inductive nonlinearity](https://doi.org/10.1063/1.4794808) uses a fast cubic root finder
developed by [Ulrich K. Deiters and Ricardo Macias-Salinas](https://doi.org/10.1021/ie4038664).