# Download
The following scripts have been assembled in order to facilitate the downloading of data required to run pyMETRIC.

### download_cdl.py
This script will download the CONUS-wide CDL image.  By default, the CDL image will be saved to the folder '.\cdl'.

### download_footprints.py
This script will download the global Landsat WRS2 descending footprint shapefile.  By default, the shapefile will be saved to the folder ".\landsat\footprints".

### download_landfire.py
LANDFIRE data will not be used for the Harney example, but the following script will download a CONUS-wide LANDFIRE image.  By default, the LANDFIRE image will be saved to the folder ".\landfire".

### download_ned.py
This script will download the 1x1 degree 1-arcsecond (~30m) resolution NED tiles that intersect the study area.  By default, the NED tiles will be saved to the folder ".\dem\tiles".  For the script to run, a shapefile of the study area extent must be provided for the "--extent" command line argument.

The NED tiles are being downloaded from the [USGS FTP server](ftp://rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/Elevation/1/IMG) and can be downloaded manually also.

### download_nlcd.py
The CONUS-wide NLCD image can be downloaded using the following command.  This script can only download the 2006 or 2011 NLCD images.  By default, the NLCD image will be saved to the folder ".\nlcd".
```
C:\pymetric>python code\download\download_nlcd.py -y 2011
```

### download_soils.py
This script will download a CONUS-wide Available Water Capacity (AWC) raster to the appropriate directory.