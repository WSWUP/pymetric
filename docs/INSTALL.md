## Installation

pyMETRIC is most easily installed by cloning the [GitHub repository](https://github.com/DRI-WSWUP/pymetric).

Most of the documentation and examples are written assuming you are running pyMETRIC on a Windows PC and that the pyMETRIC repository was cloned directly to the C: drive.  If you are using a different operating system or cloned the repository to a different location, you will need adjust commands, drive letters, and paths accordingly.

## Python

pyMETRIC has only been tested using Python 2.7 and 3.6, but may work with other versions.

## Dependencies

The following external Python modules must be present to run pyMETRIC:
* [fiona](http://toblerity.org/fiona/) (used to read and write multi-layered GIS file formats)
* [future](https://pypi.python.org/pypi/future) (adds features from Python 3 to Python 2 installations)
* [requests](http://docs.python-requests.org/en/master/) (adds enhanced http functionality)
* [scipy](https://www.scipy.org/) (provides numerous packages required for the processing of data)
* [pandas](http://pandas.pydata.org) (used to perform data processing) 
* [matplotlib](https://matplotlib.org/) (necessary for creating plots of ET related data)
* [gdal](http://www.gdal.org/) (version >2.0) (the Geospatial Data Abstraction Library is used to interact with raster and vector geospatial data)
* [netcdf4](https://www.unidata.ucar.edu/software/netcdf/) (for interacting with multi-dimensional scientific datasets, such as GRIDMET/DAYMET)
* [Landsat578](https://github.com/dgketchum/Landsat578) (for downloading Landsat images)
* [refet](https://github.com/DRI-WSWUP/RefET) (for computing reference ET)
* [drigo](https://github.com/DRI-WSWUP/drigo) (GDAL/OGR helper functions)
* [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation) (input file reader that's needed for Landsat578 downloader)

Please see the [requirements](../requirements.txt) file for details on the versioning requirements.  The module version numbers listed in the file were tested and are known to work.  Other combinations of versions may work but have not been tested.

### Python 2
The following external Python modules must be present to run pyMETRIC on Python 2
* [configparser]()(Python 2 implementation of the Python 3 configparser module)

## Anaconda/Miniconda

The easiest way of obtaining Python and all of the necessary external modules, is to install [Miniconda](https://conda.io/miniconda.html) or [Anaconda](https://www.anaconda.com/download/).

TODO: Add more explanation about where to install Miniconda (ideally to the root on the C drive) and what options need to be selected when installing.

After installing Miniconda, make sure to add the [conda-forge](https://conda-forge.github.io/) channel by entering the following in the command prompt or terminal:
```
> conda config --add channels conda-forge
```

## Conda Environment

The user is strongly encouraged to setup a dedicated conda environment for pyMETRIC:
```
> conda create -n pymetric python=3.6
```

The environment must be "activated" before use:
```
> activate pymetric
```

Most of the external modules can then be installed by calling:
```
> conda install numpy scipy pandas matplotlib gdal netcdf4 future requests yaml
```

The Landsat578 , refet, and drigo modules must be installed separately with pip:
```
> pip install Landsat578 refet drigo --no-deps
```

## Environment Variables

### Windows

#### PYTHONPATH

Many of the pyMETRIC scripts reference the "common" functions in the [pymetric/code/support](code/support) folder.  To be able to access these functions, you will need to add/append this path to the PYTHONPATH environment variable.

The environment variable can be set at the command line.  First check if PYTHONPATH is already set by typing:
```
echo %PYTHONPATH%
```
If PYTHONPATH is not set, type the following in the command prompt:
```
> setx PYTHONPATH "C:\pymetric\code\support"
```
To append to an existing PYTHONPATH, type:
```
setx PYTHONPATH "C:\pymetric\code\support;%PYTHONPATH%"
```

#### GDAL_DATA

In order to execute pyMETRIC code, the GDAL_DATA environmental variable may need to be set (*example*: GDAL_DATA = C:\Miniconda3\envs\pymetric\Library\share\gdal). **Depending on your specific installation of Python, you file path for GDAL_DATA may be different**

On a Windows PC, the user environment variables can be set through the Control Panel (System -> Advanced system settings -> Environment Variables).  Assuming that pyMETRIC was cloned/installed directly to the C: drive and Python 3 is used, the GDAL_DATA environmental variable may be set as:
```
C:\Miniconda3\envs\pymetric\Library\share\gdal
```

This environment variable can also be set at the command line.  First check if GDAL_DATA is already set by typing:
```
echo %GDAL_DATA%
```

If GDAL_DATA is not set, type the following in the command prompt:
```
> setx GDAL_DATA "C:\Miniconda3\envs\pymetric\Library\share\gdal"
```

### Mac / Linux

#### PYTHONPATH

```
echo $PYTHONPATH
```

```
export PYTHONPATH=/Users/<USER>/pymetric/code/support
```

#### GDAL_DATA

```
echo $GDAL_DATA
```

```
export GDAL_DATA=/Users/<USER>/miniconda3/envs/python3/share/gdal
```
