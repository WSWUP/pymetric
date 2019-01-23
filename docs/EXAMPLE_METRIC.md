Quicklinks: [Data Preparation](EXAMPLE_DATA.md) --- [Project Setup](EXAMPLE_SETUP.md) --- [Running METRIC](EXAMPLE_METRIC.md)

# Running METRIC
The following scripts should be ran in the following sequence in order to produce ET estimates.  These scripts are all located in the ["code/local/"](/code/local) directory.

### landsat_prep_ini.py
Prepare Landsat path/row data and populates input files to be used later in the PyMETRIC process.

### landsat_prep_scene.py
Prepares Landsat scenes for processing.

### metric_model1.py
Runs METRIC Model 1 for all images.

### metric_pixel_rating.py 
Runs METRIC pixel rating function for all images, identifying potential calibration points.

### metric_pixel_points.py
Runs METRIC pixel points function for all images, selecting initial calibration points for each Landsat image.

### metric_model2.py
Runs METRIC Model 2 for all images.

### landsat_interpolate.py
Interpolates seasonal ET data from individual METRIC scenes

# Example workflow
This workflow is setup to run with the example input file (C:\pymetric\example\landsat_2015.ini).  Use this workflow as a starting point when using pyMETRIC for your data.

```
python C:\pymetric\code\local\landsat_prep_path_row.py -i C:\pymetric\example\landsat_2015.ini
python C:\pymetric\code\local\landsat_prep_ini.py -i C:\pymetric\example\landsat_2015.ini
python C:\pymetric\code\local\landsat_prep_scene.py -i C:\pymetric\example\landsat_2015.ini
python C:\pymetric\code\local\metric_model1.py -i C:\pymetric\example\landsat_2015.ini
python C:\pymetric\code\local\metric_pixel_rating.py -i C:\pymetric\example\landsat_2015.ini
python C:\pymetric\code\local\metric_pixel_points.py -i C:\pymetric\example\landsat_2015.ini
```

__Prior to the running of METRIC model 2, calibration pixels must be adjusted manually with ArcGIS. At this point in the workflow, the software has automatically chose sample pixels, however the location of the calibration pixels must be changed for best results.  The METRIC Manual should be consulted during the calibration process in order to provide the best possible estimates of ET.  If pixels are left un-modified, ETrF rasters will still be produced, however the validity of the ETrF data will be significantly degraded.__

```
python C:\pymetric\code\local\metric_model2.py -i C:\pymetric\example\landsat_2015.ini
python C:\pymetric\code\local\landsat_interpolate.py -i C:\pymetric\example\landsat_2015.ini --tables
python C:\pymetric\code\local\landsat_interpolate.py -i C:\pymetric\example\landsat_2015.ini --rasters
```

## Running Monte Carlo Tool

The following will run one iteration of the Monte Carlo tool with fixed tail sizes of 1% (cold) and 4% (hot).  This value is the percent of agricultural pixels with ETrFs greater than the cold calibration ETrF value (for the cold calibration point).
```
python C:\pymetric\code\local\metric_monte_carlo.py -i C:\pymetric\example\landsat_2015.ini -mc 0 --tails 1 4
```

The following will run ten different iterations of the Monte Carlo tool with varying tail sizes (developed from the training data in 'misc/etrf_training_test.csv').  The "mc" parameter specifies which iterations to run.
```
python C:\pymetric\code\local\metric_monte_carlo.py -i C:\pymetric\example\landsat_2015.ini -mc 1-10
```