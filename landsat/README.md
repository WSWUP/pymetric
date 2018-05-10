# Landsat

## Landsat skip/keep lists

Before running pyMETRIC, it is important to identify Landsat images that should not be processed at all due to excessive clouds, smoke, haze, snow, shadows, or general bad data in the study area.  Many of the pyMETRIC tools are expecting or will honor a text file of Landsat scene IDs that should be skippped.  This file is typically refered to as a "skip list" in the documentation and INI files.

One approach for generating this skip list is to the use the [Cloud Free Scene Counts tools](https://github.com/Open-ET/cloud-free-scene-counts).  The Landsat path/row used in the example for those tools is 43/30 (the same path row used in the Harney example).  An [example skip list file](../harney/landsat/skip_list.txt) can be found in the Harney example folder../harney/landsat/skip_list.txt).

## Landsat Images

`Note: Landsat tar.gz files will need to be stored in nested separate folders by path, row, and year`

The Landsat images can be downloaded using the [Landsat578 tool](https://github.com/dgketchum/Landsat578).  This tool will need to be installed with pip (see the [pyMETRIC README](README)) and a credentials file will need to be generated before using (see the [Landsat 578 README](https://github.com/dgketchum/Landsat578/blob/master/README.md)).

The Landsat 7 and 8 images from 2015 for the study area can be downloaded using the following commands.  The Landsat images are being downloaded to the non-project landsat folder so that they can be used by other projects, but they could be downloaded directly to the project folder instead.
```
D:\pyMETRIC>landsat --satellite LE7 --start 2015-01-01 --end 2015-12-31 --path 43 --row 30 --output .\landsat --credentials .\landsat\usgs.txt --zipped
D:\pyMETRIC>landsat --satellite LC8 --start 2015-01-01 --end 2015-12-31 --path 43 --row 30 --output .\landsat --credentials .\landsat\usgs.txt --zipped
```

After downloading, you will need to run the following script to rename and move the Landsat tar.gz files into the correct folder structure.  Eventually, the Landsat578 download tool may support writing directly to the target folders.
```
D:\pyMETRIC>python landsat\landsat_image_organize.py
```
