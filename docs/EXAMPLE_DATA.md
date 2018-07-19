Quicklinks: [Data Preparation](EXAMPLE_DATA.md) --- [Project Setup](EXAMPLE_SETUP.md) --- [Running METRIC](EXAMPLE_METRIC.md)

# pyMETRIC Data Preparation Example

This example will step through acquiring and prepping all of the Landsat images and ancillary data needed to run the pyMETRIC code for a single Landsat path/row.  The target study area for this example is the Harney basin in central Oregon, located in Landsat path 43 row 30.

## Project Folder

All of the example script calls listed below assume that the pyMETRIC repository was installed on a Windows computer to "C:\pymetric" and that the scripts are being called from this folder.

After cloning the repository, the first step is to create a project folder if it doesn't exists.  This can be done from the command line (using the following command) or in the file explorer.
```
C:\pymetric>mkdir example
```

## Script Parameters

Most of the setup/prep scripts listed below will need to have command line arguments passed to them.  For the most part, the arguments are standardized between scripts, but the user is strongly encouraged to look at the possible arguments by first passing the help "-h" argument to the script.
For example, adding an "-h" or "--help" argument to the script call:
```
C:\pymetric>python tools\download\download_ned.py -h
```

will return the following description of the script, the possible command line arguments, as well as the argument type and default value:
```
C:\pymetric>python tools\download\download_ned.py -h
usage: download_ned.py [-h] --extent FILE [--output FOLDER] [-o] [--debug]

Download NED

optional arguments:
  -h, --help       show this help message and exit
  --extent FILE    Study area shapefile (default: None)
  --output FOLDER  Output folder (default: C:\METRIC\dem\tiles)
  -o, --overwrite  Force overwrite of existing files (default: None)
  --debug          Debug level logging (default: 20)
```

Almost all of the scripts will have the "-h", "--overwrite" (or "-o"), and "--debug" command line arguments.  The overwrite flag is used to indicate to the script that existing files should be overwritten.  If this is not set, the scripts will typically operations if the output file is present.  The debug flag is used to turn on debug level logging which will output additional text to the console and may be helpful if the scripts aren't working.

## Study Area

The first step in setting up the pyMETRIC codes is identifying or constructing a study area shapefile.  The study area shapefile path can then be passed to many of the following prep scripts using the "--extent" command line argument, in order to limit the spatial extent of the rasters.

A shapefile has been provided within this distribution to be used with the example workflow of pyMETRIC.  This example study area shapefile is located in \pymetric\example\study_area, and encompasses the area of the Harney Basin, Oregon.  This study area was derived from the USGS National Hydrography Dataset (WBDHU8).

## Landsat clear scene "keep" lists

Before running pyMETRIC, it is important to identify Landsat images that should be processed and are free of excessive clouds, smoke, haze, snow, shadows, or general bad data in the study area.  Many of the pyMETRIC tools are expecting or will honor a text file of Landsat scene IDs that should processed.  This file is typically referred to as a "keep list" in the documentation and INI files.

One approach for generating this keep list is to the use the [Cloud Free Scene Counts tools](https://github.com/DRI-WSWUP/cloud-free-scene-counts).  The Landsat path/row used in the example for those tools is also 43/30.

For the purpose of this example, we will directly use the list of clear scenes in 2015 identified at the end of the [Cloud Free Scene Counts example](https://github.com/DRI-WSWUP/cloud-free-scene-counts/blob/master/EXAMPLE.md).  The following list of 16 Landsat scene IDs should be pasted into a file called "clear_scenes.txt" and saved in "C:\pymetric\example\landsat":

```
LO08_043030_20150210
LE07_043030_20150218
LE07_043030_20150306
LC08_043030_20150415
LE07_043030_20150423
LC08_043030_20150501
LE07_043030_20150509
LE07_043030_20150610
LE07_043030_20150626
LC08_043030_20150720
LE07_043030_20150728
LE07_043030_20150813
LC08_043030_20150821
LC08_043030_20150906
LC08_043030_20150922
LE07_043030_20151016
```

## Landsat Images

`Note: Landsat tar.gz files will be stored in nested separate folders by path, row, and year`

The Landsat images can be downloaded using the [Landsat578 tool](https://github.com/dgketchum/Landsat578).  
This tool will need to be installed with pip (see the [pymetric README](README)) and a credentials file will
need to be generated before using (see the [Landsat 578 README](https://github.com/dgketchum/Landsat578/blob/master/README.md)).

```
pip install Landsat578
```

Or if you already have it, update it:

```
pip install -U Landsat578
```

The Landsat 7 and 8 images from 2015 for the study area can be downloaded using a configuration file that provides
the location of the clear_scenes.txt (the scenes determined as usable by the user, using the [Cloud Free Scene Counts tools](https://github.com/Open-ET/cloud-free-scene-counts)) and the root of the project (i.e, C:\pymetric\example).

An example configuration is located at: "C:\pymetric\example\example_downloader_config.yml"
To create a blank configuration file, please refer to the [Landsat 578 README](https://github.com/dgketchum/Landsat578/blob/master/README.md)

Example contents of the Landsat 578 configuration file:
```
# date format: 'YYYY-MM-DD'
start: '2015-01-01'
end: '2015-12-31'
path: 43
row: 30
latitude:
longitude:
output_path: C:\pymetric\landsat
satellite: 8

return_list: True
zipped: True
max_cloud_percent: 100

# pymetric directory structure: e.g. C:\pymetric\example\landsat\<path>\<row>\<year>
# using pymetric_root and clear_scenes overrides all other arguments
# leave both blank to disable

# "pymetric_root" refers to the project folder.  If this field is populated, 
# Landsat scenes will be downloaded and placed in directory structure with 
# the following format: C:\pymetric\example\landsat\<path>\<row>\<year>
# If "pymetric_root" is left blank, Landsat scenes will be downloaded into the 
# current working directory

# An unmodified config file may be generated by entering "landsat -conf example" in the command prompt window
pymetric_root: C:\pymetric\example\
clear_scenes: C:\pymetric\example\landsat\clear_scenes.txt
```

To download the Landsat scenes required for the pyMETRIC example, run the following command:
```
C:\pymetric>landsat -conf example\example_downloader_config.yml
```

This will create the directory structure pyMETRIC is expecting, e.g., 
```
C:\pymetric\example\landsat\043\030\2015\LC70430302015101LGN01.tgz
```
and download the zipped Landsat images.

### Manual Cloud Masks

Manually defined cloud mask shapefiles can be applied to each Landsat image (in addition or instead of Fmask cloud mask).  The manual cloud masks can be applied in the prep scene stage of the processing by setting the "cloud_mask_flag"  and "cloud_mask_folder" parameters in the project INI.  The cloud mask shapefiles must be named to match the Landsat image folder but with "_mask" at the end (i.d. LE07_043030_20150423_mask.shp) and must all be present in the cloud mask folder.

For this example, a sample cloud mask for image LE07_043030_20150423 is provided in the example "cloud_masks" folder.  This cloud mask was quickly drawn to exclude large portions of the image that appear to be impacted by cirrus clouds that are not being caught/flagged by Fmask.

## Ancillary Data

The ancillary data files should be downloaded once and saved in a common folder or network drive to avoid needing to repeatedly download data.

### Landsat WRS2 Descending Footprints

The following command will download the global Landsat WRS2 descending footprint shapefile.  By default, the shapefile will be saved to the folder ".\landsat\footprints".
```
C:\pymetric>python tools\download\download_footprints.py
```

### National Elevation Dataset (NED)

The following command will download the 1x1 degree 1-arcsecond (~30m) resolution NED tiles that intersect the study area.  By default, the NED tiles will be saved to the folder ".\dem\tiles".
```
C:\pymetric>python tools\download\download_ned.py --extent example\study_area\wrs2_p043r030.shp
```

The NED tiles are being downloaded from the [USGS FTP server](ftp://rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/Elevation/1/IMG) and can be downloaded manually also.

### National Land Cover Database (NLCD)

The CONUS-wide NLCD image can be downloaded using the following command.  This script can only download the 2006 or 2011 NLCD images.  By default, the NLCD image will be saved to the folder ".\nlcd".
```
C:\pymetric>python tools\download\download_nlcd.py -y 2011
```

### Cropland Data Layer (CDL) (optional)

The CDL data is updated annually and can give a slightly better representation of crop area in the study area.  CDL data can also be used in the pyMETRIC workflow to generated quasi field boundaries if a field boundary dataset is not available.  CDL data will not be used for this example, but the following command will downloaded the CONUS-wide CDL image.  By default, the CDL image will be saved to the folder '.\cdl'.

```
C:\pymetric>python tools\download\download_cdl.py -y 2015
```

### LANDFIRE (optional)

LANDFIRE data will not be used for this example, but the following command will downloaded the CONUS-wide LANDFIRE image.  By default, the LANDFIRE image will be saved to the folder ".\landfire".

```
C:\pymetric>python tools\download\download_landfire.py -v 140
```

### Available Water Capacity (AWC)

CONUS-wide AWC rasters can be downloaded to the appropriate directory using the following script:
```
C:\pymetric>python tools\download\download_soils.py
```
CONUS-wide AWC rasters can be manually downloaded from the following URLs:
* STATSGO - [https://storage.googleapis.com/openet/statsgo/AWC_WTA_0to10cm_statsgo.tif](https://storage.googleapis.com/openet/statsgo/AWC_WTA_0to10cm_statsgo.tif)
* SSURGO - [https://storage.googleapis.com/openet/ssurgo/AWC_WTA_0to10cm_composite.tif](https://storage.googleapis.com/openet/ssurgo/AWC_WTA_0to10cm_composite.tif)

## Daily Weather Data

Weather data are stored in multi-band rasters with a separate band for each day of year (DOY).  This was primarily done to reduce the total number of files generated but also helps simplify the data extraction within the Python code.

### GRIDMET

Generate elevation, latitude, and longitude rasters.
```
C:\pymetric>python tools\gridmet\gridmet_ancillary.py
```

The following command will download the precipitation (PPT) and reference ET (ETr) components variable NetCDF files.  Make sure to always download and prep a few extra months of data before the target date range in order have enough extra to spin-up the soil water balance.
```
C:\pymetric>python tools\gridmet\gridmet_download.py --start 2014-10-01 --end 2015-12-31
```

The following commands will generate daily reference ET (from the components variables) and precipitation IMG rasters.
```
C:\pymetric>python tools\gridmet\gridmet_daily_refet.py --start 2014-10-01 --end 2015-12-31 --extent example\study_area\wrs2_p043r030.shp
C:\pymetric>python tools\gridmet\gridmet_daily_ppt.py --start 2014-10-01 --end 2015-12-31 --extent example\study_area\wrs2_p043r030.shp
```

### Spatial CIMIS

Generate elevation, latitude, and longitude rasters.
```
C:\pymetric>python tools\cimis\cimis_ancillary.py
```

The following command will download the reference ET (ETr) components variable GZ files.  Make sure to always download and prep a few extra months of data before the target date range in order to spin-up the soil water balance.
```
C:\pymetric>python tools\cimis\cimis_download.py --start 2014-10-01 --end 2015-12-31
```

The ASCII rasters then need to be extracted from the GZ files and converted to IMG.
```
C:\pymetric>python tools\cimis\cimis_extract_convert.py --start 2014-10-01 --end 2015-12-31
```

The following commands will generate daily reference ET (from the components variables)
```
C:\pymetric>python tools\cimis\cimis_daily_refet.py --start 2014-10-01 --end 2015-12-31 --extent example\study_area\wrs2_p043r030.shp
```

GRIDMET (or anothere data set) must still be used for the precipitation, since it is not provided with Spatial CIMIS.
```
C:\pymetric>python tools\gridmet\gridmet_download.py --start 2014-10-01 --end 2015-12-31 --vars pr
C:\pymetric>python tools\gridmet\gridmet_daily_ppt.py --start 2014-10-01 --end 2015-12-31 --extent example\study_area\wrs2_p043r030.shp
```

## Hourly Weather Data

### NLDAS

```
C:\pymetric>python tools\nldas\nldas_ancillary.py
```

In order to download the NLDAS hourly data, you will need to create an [Earthdata login](https://urs.earthdata.nasa.gov/).  Once your account is created with Earthdata, data access to Goddard Earth Sciences Data and Information Services Center (GES DISC) must be enabled for your account.  To approve access:
1. [Navigate to your Earthdata profile page](https://urs.earthdata.nasa.gov/)
2. Select "Applications -> Authorized Apps"
3. Click the "APPROVE MORE APPLICATIONS" button
4. Approve "NASA GESDISC DATA ARCHIVE"

**(For descriptive instructions, please visit: [https://disc.gsfc.nasa.gov/earthdata-login](https://disc.gsfc.nasa.gov/earthdata-login))**

Begin downloading the NLDAS hourly GRB files.  All of the NLDAS variables for a single hour are stored in a single GRB file.  The "--landsat" parameter is set in order to limit the download to only those dates and times that are needed for the Landsat images in the study area and time period.  If you don't specify the "--landsat" parameter, the script will attempt to download all hourly data within the "--start" and "--end" range.

```
C:\pymetric>python tools\nldas\nldas_download.py <USERNAME> <PASSWORD> --start 2015-01-01 --end 2015-12-31  --landsat example\landsat\clear_scenes.txt
```

#### Reference ET (ETr)

This code also supports the processing of both hourly ETo (Grass reference evapotranspiration) and ETr (Alfalfa reference evapotranspiration).  For the purposes of pyMETRIC, only ETr is needed.

The "--landsat" argument is optional at this point, since GRB files were only downloaded for Landsat dates in the previous step.  This flag can be useful for other projects if you have downloaded a more complete set of NLDAS data.

```
C:\pymetric>python tools\nldas\nldas_hourly_refet.py --start 2015-01-01 --end 2015-12-31 --extent example\study_area\wrs2_p043r030.shp --landsat example\landsat\clear_scenes.txt
```

#### Vapor Pressure

```
C:\pymetric>python tools\nldas\nldas_hourly_ea.py --start 2015-01-01 --end 2015-12-31 --extent example\study_area\wrs2_p043r030.shp --landsat example\landsat\clear_scenes.txt
```

#### Wind Speed

```
C:\pymetric>python tools\nldas\nldas_hourly_wind.py --start 2015-01-01 --end 2015-12-31 --extent example\study_area\wrs2_p043r030.shp --landsat example\landsat\clear_scenes.txt
```

#### Additional Parameters

#### Optimization
:red_circle: Explain some of the NLDAS command line arguments

+ stats: Compute raster statistics
+ times: To minimize the amount of data that needs to be downloaded and stored in each daily file, the following three scripts can all be run with a "--times" argument to specify which hours to process.
+ te:  To minimize the amount of data that needs to be downloaded and stored in each daily file, a custom extent can be manually entered.  This argument requires the input of a western limit, southern limit, eastern limit, and northern limit (x-min, y-min, x-max, and y-max) in units of decimal degrees.


# Command Summary

Below is an example work flow for downloading all ancillary data needed to run pyMETRIC for the Harney Basin study area example.  Be aware that this is only an example and that variations in your installation directory or general setup may render these commands inoperable.  Provided that pyMETRIC has been installed in the C: directory, these commands or formatted so that they may be pasted into the Windows Command Prompt *(accessible by pressing 'Windows Key'+R on your keyboard and typing "CMD" in the 'Run' window)*.

#### Download gridMET data

This section will perform the downloading and processing of daily meteorological data necessary for calculating and interpolating evapotranspiration estimates, and must be downloaded for the entire period of interest.  For the purposes of this example, data is acquired for the entire year of 2015.

```
python C:\pymetric\tools\gridmet\gridmet_ancillary.py
python C:\pymetric\tools\gridmet\gridmet_download.py --start 2015-01-01 --end 2015-12-31
python C:\pymetric\tools\gridmet\gridmet_daily_refet.py --start 2015-01-01 --end 2015-12-31 -o -te -120.1 42.4 -118.3 44.3 --netcdf C:\pymetric\gridmet\netcdf
python C:\pymetric\tools\gridmet\gridmet_daily_temp.py --start 2015-01-01 --end 2015-12-31 -o -te -120.1 42.4 -118.3 44.3 --netcdf C:\pymetric\gridmet\netcdf
python C:\pymetric\tools\gridmet\gridmet_daily_ppt.py --start 2015-01-01 --end 2015-12-31 -o -te -120.1 42.4 -118.3 44.3 --netcdf C:\pymetric\gridmet\netcdf
python C:\pymetric\tools\gridmet\gridmet_daily_ea.py --start 2015-01-01 --end 2015-12-31 -o -te -120.1 42.4 -118.3 44.3 --netcdf C:\pymetric\gridmet\netcdf
```

#### Download NLDAS data

This section will perform the downloading and processing of hourly meteorological data necessary for calculating and interpolating evapotranspiration estimates, and must be downloaded for the entire period of interest.  For the purposes of this example, data is acquired for the entire year of 2015.

__Please note that that an [Earthdata username and password](https://urs.earthdata.nasa.gov/) must be acquired in order to download NLDAS data.__

```
python C:\pymetric\tools\nldas\nldas_ancillary.py
python C:\pymetric\tools\nldas\nldas_download.py <Earthdata USERNAME> <Earthdata PASSWORD> --start 2015-01-01 --end 2015-12-31  --landsat example\landsat\clear_scenes.txt
python C:\pymetric\tools\nldas\nldas_hourly_ea.py --start 2015-01-01 --end 2015-12-31 -te -120.1 42.4 -118.3 44.3 --grb C:\pymetric\nldas\grb
python C:\pymetric\tools\nldas\nldas_hourly_refet.py --start 2015-01-01 --end 2015-12-31 -te -120.1 42.4 -118.3 44.3 --grb C:\pymetric\nldas\grb
python C:\pymetric\tools\nldas\nldas_hourly_wind.py --start 2015-01-01 --end 2015-12-31 -te -120.1 42.4 -118.3 44.3 --grb C:\pymetric\nldas\grb
```
#### Download Landcover/Land surface data

This section downloads land surface data. This data includes information on elevation, agricultural land delineation, land cover, and Landsat footprints.

```
python C:\pymetric\tools\download\download_footprints.py
python C:\pymetric\tools\download\download_ned.py --extent C:\pymetric\example\study_area\wrs2_p043r030.shp
python C:\pymetric\tools\download\download_cdl.py --year 2015
python C:\pymetric\tools\download\download_landfire.py
python C:\pymetric\tools\download\download_nlcd.py --year 2011
```

#### Landsat data download and prep

pyMETRIC uses the [Landsat578](https://github.com/dgketchum/Landsat578) package for downloading Landsat imagery products.
```
C:\pymetric>landsat -conf example\example_downloader_config.yml
```
