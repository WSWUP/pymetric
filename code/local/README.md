pyMETRIC Workflow
=================

>**Note:** 
>pyMETRIC makes use of the [ArgParse](https://docs.python.org/3/library/argparse.html) module which enables the use of command-line options, arguments and sub-commands.  This functionality is only available when running pyMETRIC through the command line.  Operating pyMETRIC through the graphic user interface (GUI) of your operating system is not recommended.

Input File
-------------
Components of pyMETRIC require the use of an input file (.ini), which sets parameters to be used in processing.  The pathc to the input file must be preceded by "-i" or "--ini".  (Example: -i C:\pymetric\example\landsat_2015.ini)

 
Common Flags
-------------
> -h : triggers a 'help' function within the interface, which will list available arguments for the code being operated 
> -o : indicates that output data from previous pyMETRIC runs will be overwritten by the execution of the code
> -mp : allows for 'multiprocessing' of code, where multiple CPU cores will be used to run iterations of the process concurrently
> -i : path to input file (.ini), where inputs for code execution are stored (this flag is mandatory for most scripts within pyMETRIC)


Data Preparation
-------------
### landsat_prep_path_row.py
**-i** or **-\-ini (str)**
:   File path of the input parameters file

**-o** or **-\-overwrite_flag (bool)**: 
:   If True, overwrite existing files

**-mp (int)** or **-\-multiprocessing (int)**: 
:   Number of cpu cores to use for processing

**-d** or **-\-debug (bool)**:
:   If True, enable debug level logging

### landsat_prep_ini.py
**ini_path (str)**
:   File path of the input parameters file

**-o** or **-\-overwrite_flag (bool)**: 
:   If True, overwrite existing files

**-d** or **-\-debug (bool)**:
:   If True, enable debug level logging

**-mp (int)** or **-\-multiprocessing (int)**: 
:   Number of cpu cores to use for processing

**-\-no_smooth (int)**: 
:   Don't dilate and erode image to remove fringe/edge pixels

**-\-no_stats (bool)**: 
:   If True, compute raster statistics

**-o** or **-\-overwrite_flag (bool)**: 
:   If True, overwrite existing files

**-pr** or **-\-path_row (str)**
:   Landsat path/rows to process (pXXrYY)

**-\-window (bool)**: 
:   If True, each process will be opened in a new terminal

### landsat_prep_scene.py
**-i** or **-\-ini (str)**
:   File path of the input parameters file

**-bs** or **-\-blocksize (str)**
:   Block size

**-\-delay (float)**: 
:   Max random delay starting function in seconds


METRIC
-------------
### metric_model1.py
**-i** or **-\-ini (str)**
:   File path of the input parameters file

**-d** or **-\-debug (bool)**:
:   If True, enable debug level logging

**-\-delay (float)**: 
:   Max random delay starting function in seconds

**-mp (int)** or **-\-multiprocessing (int)**: 
:   Number of cpu cores to use for processing

**-\-no_stats (bool)**: 
:   If True, compute raster statistics

**-o** or **-\-overwrite_flag (bool)**: 
:   If True, overwrite existing files

**-pr** or **-\-path_row (str)**
:   Landsat path/rows to process (pXXrYY)

**-\-window (bool)**: 
:   If True, each process will be opened in a new terminal

### metric_pixel_rating.py
**-i** or **-\-ini (str)**
:   File path of the input parameters file

**-d** or **-\-debug (bool)**:
:   If True, enable debug level logging

**-\-delay (float)**: 
:   Max random delay starting function in seconds

**-mp (int)** or **-\-multiprocessing (int)**: 
:   Number of cpu cores to use for processing

**-\-no_stats (bool)**: 
:   If True, compute raster statistics

**-o** or **-\-overwrite_flag (bool)**: 
:   If True, overwrite existing files

**-pr** or **-\-path_row (str)**
:   Landsat path/rows to process (pXXrYY)

**-\-window (bool)**: 
:   If True, each process will be opened in a new terminal

### metric_pixel_points.py
**-i** or **-\-ini (str)**
:   File path of the input parameters file

**-gs** or **-\-groupsize (str)**
:   Minimum group size for placing calibration points

**-bs** or **-\-blocksize (str)**
:   Block size

**-\-no_shapefile (str)**
:   Don't save calibration points to shapefile

**-j** or **-\-geojson (str)**
:   Don't save calibration points to shapefile

**-o** or **-\-overwrite_flag (bool)**: 
:   If True, overwrite existing files

**-mp (int)** or **-\-multiprocessing (int)**: 
:   Number of cpu cores to use for processing

**-\-window (bool)**: 
:   If True, each process will be opened in a new terminal

**-\-delay (float)**: 
:   Max random delay starting function in seconds

**-d** or **-\-debug (bool)**:
:   If True, enable debug level logging

### metric_model2.py
**-i** or **-\-ini (str)**
:   File path of the input parameters file

**-d** or **-\-debug (bool)**:
:   If True, enable debug level logging

**-\-delay (float)**: 
:   Max random delay starting function in seconds

**-mp (int)** or **-\-multiprocessing (int)**: 
:   Number of cpu cores to use for processing

**-\-no_stats (bool)**: 
:   If True, compute raster statistics

**-o** or **-\-overwrite_flag (bool)**: 
:   If True, overwrite existing files

**-pr** or **-\-path_row (str)**
:   Landsat path/rows to process (pXXrYY)

**-\-window (bool)**: 
:   If True, each process will be opened in a new terminal

### landsat_interpolate.py
**-i** or **-\-ini (str)**
:   File path of the input parameters file

**-d** or **-\-debug (bool)**:
:   If True, enable debug level logging

**-\-delay (float)**: 
:   Max random delay starting function in seconds

**-mp (int)** or **-\-multiprocessing (int)**: 
:   Number of cpu cores to use for processing

**-\-no_pyramids (bool)**: 
:   If True, compute raster pyramids

**-\-no_stats (bool)**: 
:   If True, compute raster statistics

**-\-no_file_logging (bool)**: 
:   If True, don't write logging to file

**-o** or **-\-overwrite_flag (bool)**: 
:   If True, overwrite existing files

**-pr** or **-\-path_row (str)**
:   Landsat path/rows to process (pXXrYY)

**-\-rasters (bool)**
:   If True, override INI and interpolate rasters

**-\-tables (bool)**
:   If True, override INI and interpolate zone tables

## Example sequence of running the python codes within this directory
__Please note that these scripts will only run effectively if the required data is downloaded and structured as prescribed in the [Example Data readme](../../docs/EXAMPLE_DATA.md).__
```
python C:\pymetric\code\local\landsat_prep_path_row.py -i C:\pymetric\example\landsat_2015.ini
python C:\pymetric\code\local\landsat_prep_ini.py -i C:\pymetric\example\landsat_2015.ini
python C:\pymetric\code\local\landsat_prep_scene.py -i C:\pymetric\example\landsat_2015.ini -mp
python C:\pymetric\code\local\metric_model1.py -i C:\pymetric\example\landsat_2015.ini -mp
python C:\pymetric\code\local\metric_pixel_rating.py -i C:\pymetric\example\landsat_2015.ini -mp
python C:\pymetric\code\local\metric_pixel_points.py -i C:\pymetric\example\landsat_2015.ini -mp
python C:\pymetric\code\local\metric_model2.py -i C:\pymetric\example\landsat_2015.ini -mp
python C:\pymetric\code\local\landsat_interpolate.py -i C:\pymetric\example\landsat_2015.ini -mp --tables
python C:\pymetric\code\local\landsat_interpolate.py -i C:\pymetric\example\landsat_2015.ini -mp --rasters
```
