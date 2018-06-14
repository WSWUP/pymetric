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
This workflow is setup to run with the example input file (D:\pymetric\example\landsat_2015.ini).  Use this workflow as a starting point when using pyMETRIC for your data.

```
python D:\pymetric\code\local\landsat_prep_path_row.py -i D:\pymetric\example\landsat_2015.ini
python D:\pymetric\code\local\landsat_prep_ini.py -i D:\pymetric\example\landsat_2015.ini
python D:\pymetric\code\local\landsat_prep_scene.py -i D:\pymetric\example\landsat_2015.ini
python D:\pymetric\code\local\metric_model1.py -i D:\pymetric\example\example.ini
python D:\pymetric\code\local\metric_pixel_rating.py -i D:\pymetric\example\landsat_2015.ini
python D:\pymetric\code\local\metric_pixel_points.py -i D:\pymetric\example\landsat_2015.ini
```

__Prior to the running of METRIC model 2, calibration pixels must be adjusted manually with ArcGIS. At this point in the workflow, the software has automatically chose sample pixels, however the location of the calibration pixels must be changed for best results.  The METRIC Manual should be consulted during the calibration process in order to provide the best possible estimates of ET.  If pixels are left un-modified, ETrF rasters will still be produced, however the validity of the ETrF data will be significantly degraded.__

```
python D:\pymetric\code\local\metric_model2.py -i D:\pymetric\example\landsat_2015.ini
python D:\pymetric\code\local\landsat_interpolate.py -i D:\pymetric\example\landsat_2015.ini --tables
python D:\pymetric\code\local\landsat_interpolate.py -i D:\pymetric\example\landsat_2015.ini --rasters
```