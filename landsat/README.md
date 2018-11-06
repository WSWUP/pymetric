# Landsat

## Landsat clear scene "keep" lists

Before running pyMETRIC, it is important to identify Landsat images that should be processed and are free of excessive clouds, smoke, haze, snow, shadows, or general bad data in the study area.  Many of the pyMETRIC tools are expecting or will honor a text file of Landsat scene IDs that should processed.  This file is typically referred to as a "keep list" in the documentation and INI files.

One approach for generating this keep list is to the use the [Cloud Free Scene Counts tools](https://github.com/DRI-WSWUP/cloud-free-scene-counts).  The Landsat path/row used in the example for those tools is also 43/30.  An [example keep list file](../example/landsat/clear_scenes.txt) can be found in the Harney example folder.

## Landsat Images

The following command will download the Landsat scenes required for the pyMETRIC example.  The start and end date parameters are only needed if the clear scene list includes scenes from other years.  The Landsat images are being downloaded to the non-project landsat folder so that they can be used by other projects, but they could be downloaded directly to the project folder instead.
```
C:\pymetric>python tools\download\download_landsat.py example\landsat\clear_scenes.txt --start 2015-01-01 --end 2015-12-31
```

This will create the directory structure pyMETRIC is expecting, with tar.gz files will be stored in nested separate folders by path, row, and year:

```
C:\pymetric\example\landsat\043\030\2015\LC70430302015101LGN01.tgz
```
