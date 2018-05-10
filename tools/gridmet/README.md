# GRIDMET

Scripts for downloading and preparing GRIDMET daily weather data.  

### gridmet_ancillary.py
-------------
Download and process the GRIDMET elevation, latitude, and longitude rasters

**-o** or **-\-overwrite_flag (bool)**: 
:   If True, overwrite existing files

**-d** or **-\-debug (bool)**:
:   If True, enable debug level logging

### gridmet_download.py - Download the ".nc" files from the GRIDMET website
-------------
The default date range is 2017-01-01 to 2017-12-31

### gridmet_daily_refet.py
-------------
- Calculate daily ETo and ETr from the GRIDMET inputs. ETo and ETr are saved as daily IMG rasters in separate folders

### gridmet_daily_ppt.py
-------------
- Calculate daily precipitation from the GRIDMET inputs

# These are optional scripts for extracting other GRIDMET variables

### gridmet_daily_temp.py
-------------
- Calculate daily min/max temperature from the GRIDMET inputs

### gridmet_daily_ea.py
-------------
- Calculate daily vapor pressure from the GRIDMET inputs

### gridmet_daily_variables.py
-------------
- Calculate daily variables from the GRIDMET inputs
