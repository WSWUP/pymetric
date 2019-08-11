# ===============================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import os

home = os.path.expanduser('~')
PATH = os.path.join(home, 'miniconda2', 'envs', 'qgis', 'share', 'qgis', 'python')
import sys

sys.path.append(PATH)

import os
from qgis.core import QgsApplication, QgsProject, QgsMapLayerRegistry
from qgis.core import QgsRasterLayer, QgsVectorLayer
from PyQt4.QtCore import QFileInfo


def make_qgs(qml, input):
    shapes = {'cold': os.path.join(input, 'PIXELS', 'cold.shp'),
              'hot': os.path.join(input, 'PIXELS', 'hot.shp')}

    rasters = {'hot_pixels': os.path.join(input, 'PIXEL_REGIONS', 'hot_pixel_suggestion.img'),
               'cold_pixels': os.path.join(input, 'PIXEL_REGIONS', 'cold_pixel_suggestion.img'),
               'region_mask': os.path.join(input, 'PIXEL_REGIONS', 'region_mask.img'),
               'ndvi': os.path.join(input, 'INDICES', 'ndvi_toa.img'),
               'ts': os.path.join(input, 'ts.img'),
               'albedo': os.path.join(input, 'albedo_at_sur.img'),
               'et_rf': os.path.join(input, 'ETRF', 'et_rf.img')}

    keys = ['cold', 'hot', 'hot_pixels', 'cold_pixels', 'region_mask', 'ndvi', 'ts', 'albedo', 'et_rf']

    QgsApplication.setPrefixPath(PATH, True)
    qgs = QgsApplication([], False)
    qgs.initQgis()

    project = QgsProject.instance()
    project.read(QFileInfo(os.path.join(qml, 'qgis_template.qgs')))
    project_path = QFileInfo(os.path.join(input, 'calbration_map.qgs'))
    for key, path in rasters.items():
        layer = QgsRasterLayer(path)
        if not layer.isValid():
            print("file {} is not a valid raster file".format(path))
        QgsMapLayerRegistry.instance().addMapLayer(layer)

    for key, path in shapes.items():
        fileInfo = QFileInfo(path)
        path = fileInfo.filePath()
        baseName = fileInfo.baseName()
        layer = QgsVectorLayer(path, baseName, 'ogr')
        if not layer.isValid():
            print("file {} is not a valid vector file".format(path))

        QgsMapLayerRegistry.instance().addMapLayer(layer)

    for layer, name in zip(QgsMapLayerRegistry.instance().mapLayers().values(), keys):
        layer.loadNamedStyle(os.path.join(qml, '{}{}'.format(qml, '.qml')))

    project.write(project_path)
    qgs.exitQgis()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    path = os.path.join(home, 'IrrigationGIS', 'tests', 'qgis')
    folder = os.path.join(path, 'LC08_041027_20150807')
    make_qgs(path, folder)
# ========================= EOF ====================================================================
