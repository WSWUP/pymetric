# CIMIS

Scripts for downloading and preparing CIMIS daily weather data

cimis_ancillary.py - Download and process the CIMIS mask, elevation, 
    latitude, and longitude rasters
cimis_download.py - Download the ".asc.gz" files from the CIMIS website
    The default date range is 2017-01-01 to 2017-12-31
cimis_extract_convert.py - Uncompress the ".asc.gz" files and convert to IMG
cimis_daily_refet.py - Calculate daily ETo and ETr from the CIMIS inputs
    ETo and ETr are saved as daily IMG rasters in separate folders
cimis_nldas_fill.py - Get NLDAS 4km daily ETo/ETr for dates with missing data

CIMIS elevation data is from:
    Global Multi-resolution Terrain Elevation Data 2010 (GMTED2010)
    http://topotools.cr.usgs.gov/gmted_viewer/
    http://topotools.cr.usgs.gov/gmted_viewer/data/Grid_ZipFiles/mn30_grd.zip
    http://topotools.cr.usgs.gov/gmted_viewer/data/Grid_ZipFiles/md30_grd.zip