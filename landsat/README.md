# Landsat

## Landsat clear scene "keep" lists

Before running pyMETRIC, it is important to identify Landsat images that should be processed and are free of excessive clouds, smoke, haze, snow, shadows, or general bad data in the study area.  Many of the pyMETRIC tools are expecting or will honor a text file of Landsat scene IDs that should processed.  This file is typically referred to as a "keep list" in the documentation and INI files.

One approach for generating this keep list is to the use the [Cloud Free Scene Counts tools](https://github.com/DRI-WSWUP/cloud-free-scene-counts).  The Landsat path/row used in the example for those tools is also 43/30.  An [example keep list file](../example/landsat/clear_scenes.txt) can be found in the Harney example folder../example/landsat/clear_scenes.txt).

## Landsat Images

`Note: Landsat tar.gz files will need to be stored in nested separate folders by path, row, and year`

The Landsat images can be downloaded using the [Landsat578 tool](https://github.com/dgketchum/Landsat578).  This tool will need to be installed with pip (see the [pyMETRIC README](README)).

The Landsat 7 and 8 images from 2015 for the study area can be downloaded using the following commands.  The Landsat images are being downloaded to the non-project landsat folder so that they can be used by other projects, but they could be downloaded directly to the project folder instead.
```
D:\pymetric>landsat -conf example\example_downloader_config.yml
```
