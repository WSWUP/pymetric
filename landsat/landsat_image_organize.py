from builtins import input
import datetime
import os
import re
import shutil

workspace = os.getcwd()

folder_re = re.compile(
    '^(?P<satellite>L[TECO][4578])_(?P<path>\d+)_(?P<row>\d+)$')
pre_re = re.compile(
    '^(?P<satellite>L[TECO][4578])(?P<path>\d{3})(?P<row>\d{3})'
    '(?P<year>\d{4})(?P<doy>\d{3})(?P<station>\w{3})(?P<version>\d{2})$')
# c1_re = re.compile(
#     '^(??P<satellite>L[TECO]0[4578])_(?P<path>\d{3})(?P<row>\d{3})_'
#     '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})$')

# This would be a little cleaner using os.walk()
for input_folder in os.listdir(workspace):
    input_ws = os.path.join(workspace, input_folder)
    input_match = folder_re.match(input_folder)
    if not input_match:
        continue
    # elif not os.path.isdir(input_ws):
    #     continue
    print(input_folder)

    # Iterate through image folders
    for image_folder in os.listdir(input_ws):
        image_ws = os.path.join(input_ws, image_folder)
        image_match = pre_re.match(image_folder)
        if not image_match:
            continue
        # elif not os.path.isdir(image_ws):
        #     continue
        print('  ' + image_folder)

        # Unpack the input ID components
        satellite = image_match.group('satellite')
        satellite = satellite[:2] + '0' + satellite[2]
        path = int(image_match.group('path'))
        row = int(image_match.group('row'))
        year = int(image_match.group('year'))
        doy = int(image_match.group('doy'))
        image_dt = datetime.datetime.strptime(
            '{:04d}_{:03d}'.format(year, doy), '%Y_%j')

        # Compute the EE scene ID
        output_id = '{}_{:03d}{:03d}_{}'.format(
            satellite, path, row, image_dt.strftime('%Y%m%d'))

        # Copy and rename the tar gz
        input_path = os.path.join(image_ws, image_folder + '.tgz')
        output_path = os.path.join(
            workspace, str(path), str(row), str(year), output_id + '.tar.gz')

        if not os.path.isdir(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        if os.path.isfile(input_path) and not os.path.isfile(output_path):
            shutil.copy(input_path, output_path)
            # shutil.move(input_path, output_path)

