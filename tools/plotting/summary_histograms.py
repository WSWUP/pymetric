import argparse
import logging
import datetime as dt
import os
import sys
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from scipy import stats

import python_common as dripy


def main(csv_path, output_folder, fid_list='', bin_min=0, bin_max=5,
         bin_size=0.25, start_dt=None, end_dt=None, plots='all'):

    """Create Summary Histogram Plots from pymetric zonal csv output files
    Args:
        csv_path (str): zonal stats file path
        output_folder (str): Folder path where files will be saved
                            default(...pymetric/summary_histograms)
        fid_list (list): list or range of FIDs to skip
        bin_min (int): Histogram Minimum (default: 0)
        bin_max (int): Histogram Max (default: 5)
        bin_size (int): Histogram bin size (default: 0.25)
        start_dt : datetime (start date; optional)
        end_dt : datetime (end date; optional)
        plots (str): Output plot options: all, acreage, or field (default: all)
    Returns:
        None
    """
    logging.info('\nReading input csv file: {}'.format(csv_path))

    # Check if csv file exist
    if not csv_path:
        logging.error(
            'ERROR: csv file does not exist')
        sys.exit()
    # Attempt to read csv_file
    try:
        input_df = pd.read_csv(csv_path, sep=',')
    except:
        logging.error('Error reading file. Check csv path.')
        sys.exit()

    # Create Output Folder if it doesn't exist
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Filter FIDs based on fid_list (default [])
    fid_skiplist = []
    if fid_list:
        fid_skiplist = sorted(list(dripy.parse_int_set(fid_list)))
        logging.info('Skipping FIDs: {}'.format(fid_skiplist))
        input_df = input_df[~input_df['FID'].isin(fid_skiplist)]

    if (start_dt and not end_dt) or (end_dt and not start_dt):
        logging.error('\nPlease Specify Both Start and End Date:'
                      '\nStart Date: {}'
                      '\nEnd Date: {}'.format(start_dt, end_dt))
        sys.exit()

    if (start_dt and end_dt) and (end_dt < start_dt):
        logging.error('End date cannot be before start date.'
                      ' Exiting.')
        sys.exit()

    # Filter dataset if start and end dates are specified
    if start_dt and end_dt:
        if 'DATE' in input_df.columns:
            input_df['DATE'] = pd.to_datetime(input_df['DATE'])
            logging.info('\nFiltering By Date. Start: {:%Y-%m-%d}, '
                         'End: {:%Y-%m-%d}'.format(start_dt, end_dt))
            input_df = input_df[(input_df['DATE'] >= start_dt) &
                                (input_df['DATE'] <= end_dt)]
            if input_df.empty:
                logging.error('Date Filter Removed All Data. Exiting.')
                sys.exit()
        else:
            logging.error('Cannot Apply Custom Date Range On Monthly OR Annual'
                          ' Datasets. \nUse Daily Output. Exiting.')
            sys.exit()

    # Unit Conversions
    pix2acre = 0.222395  # 30x30m pixel to acres; From Google
    mm2ft = 0.00328084  # From Google

    # Add Acres
    input_df['Area_acres'] = input_df.PIXELS * pix2acre
    # Add FT Fields
    input_df['ET_FT'] = input_df.ET_MM * mm2ft
    input_df['ETR_FT'] = input_df.ETR_MM * mm2ft
    # Daily Volume Field
    input_df['Volume_acft'] = input_df.Area_acres * input_df.ET_FT
    # Net ET Field
    input_df['NetET_mm'] = input_df.ET_MM - input_df.PPT_MM
    input_df['NetET_FT'] = input_df.NetET_mm * mm2ft
    input_df['NetVolume_acft'] = input_df.Area_acres * input_df['NetET_FT']

    # Growing Season Start/End Months (inclusive)
    start_month = 4
    end_month = 10

    # Create Growing Season Only Dataframe
    if 'MONTH' in input_df.columns:
        gs_df = input_df[(input_df['MONTH'] >= start_month) &
                         (input_df['MONTH'] <= end_month)]

    # Dictionary to control agg of each variable
    a = {'FID': 'mean',
         'YEAR': 'mean',
         'PIXELS': 'mean',
         'NDVI': 'mean',
         'ETRF': 'mean',
         'ETR_MM': 'sum',
         'ET_MM': 'sum',
         'PPT_MM': 'sum',
         'Area_acres': 'mean',
         'ET_FT': 'sum',
         'ETR_FT': 'sum',
         'NetET_FT': 'sum',
         'Volume_acft': 'sum',
         'NetVolume_acft': 'sum'}

    # GS Grouped Dataframe (only for daily and monthly csv)
    if 'MONTH' in input_df.columns:
        gs_grp_df = gs_df.groupby('FID', as_index=True).agg(a)

    # Annual Grouped Dataframe
    ann_grp_df = input_df.groupby('FID', as_index=True).agg(a)

    # Field Count Histogram Function
    def field_count_hist(grp_df, rate_var, vol_var, title, xlab, filedesc):
        # Annotation Box Stats
        y = grp_df['YEAR'].mean()
        total_area = grp_df['Area_acres'].sum()
        total_vol = grp_df[vol_var].sum()
        m = total_vol/total_area.round(1)

        # Bins
        et_bins = np.linspace(bin_min, bin_max,
                              ((bin_max-bin_min) / bin_size) + 1)

        # Make Figure
        font_size = 12
        ann_font_size = 10
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(grp_df[rate_var], bins=et_bins, align='mid', edgecolor='black')
        ax.set_title(title, size=font_size)
        ax.set_xlabel(xlab, size=font_size)
        ax.set_ylabel('Field Count', size=font_size)
        ax.set_xticks(np.arange(0, bin_max + (2*bin_size), 2*bin_size))
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        ymin, ymax = plt.ylim()  # return the current ylim
        plt.ylim((ymin, ymax + ymax * 0.3))  # shift ymax for annotation space
        # Add mean vertical line
        ax.axvline(m, color='gray', linestyle='dashed',
                   linewidth=1)

        # Add Annotation Text Box
        antext = (
                'Year {:.0f}\n' +
                'Mean ET = {:.1f} ft\n' +
                'Total Area = {:.1f} acres\n' +
                'ET Volume = {:.1f} ac-ft').format(
            y, m, total_area, total_vol)
        at = AnchoredText(
            antext, prop=dict(size=ann_font_size), frameon=True, loc=2)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        # Save Figure
        file_name = '{:.0f}_{}_Fields'.format(y, filedesc)
        fig.tight_layout(pad=3)
        plt.savefig(os.path.join(output_folder, file_name), dpi=300)
        plt.close(fig)
        fig.clf()
        return True

    # Acreage histogram (Bar Plot)
    def acreage_histogram(grp_df, rate_var, vol_var, title, xlab, filedesc):
        # Annotation Box Stats
        y = grp_df['YEAR'].mean()
        total_area = grp_df['Area_acres'].sum()
        total_vol = grp_df[vol_var].sum()
        m = total_vol/total_area.round(1)

        # Bins
        et_bins = np.linspace(bin_min, bin_max,
                              ((bin_max - bin_min) / bin_size) + 1)

        # Acreage/ET Bins
        et_area_hist, et_bins, binnum = stats.binned_statistic(
            grp_df[rate_var], grp_df.Area_acres, 'sum', et_bins)

        # Make Figure
        font_size = 12
        ann_font_size = 10
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(et_bins[:-1], et_area_hist, width=bin_size, edgecolor='black',
               align='edge', color='r')
        ax.set_title(title, size=font_size)
        ax.set_xlabel(xlab, size=font_size)
        ax.set_ylabel('Acreage', size=font_size)
        ax.set_xticks(np.arange(0, bin_max + (2*bin_size), 2*bin_size))
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        ymin, ymax = plt.ylim()  # return the current ylim
        plt.ylim((ymin, ymax + ymax*0.3))  # shift ymax for annotation space
        # Add mean vertical line
        ax.axvline(m, color='gray', linestyle='dashed',
                   linewidth=1)

        # Add Annotation Text Box
        antext = (
                'Year {:.0f}\n' +
                'Mean ET = {:.1f} ft\n' +
                'Total Area = {:.1f} acres\n' +
                'ET Volume = {:.1f} ac-ft').format(
            y, m, total_area, total_vol)
        at = AnchoredText(
            antext, prop=dict(size=ann_font_size), frameon=True, loc=2)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        # Save Figure
        file_name = '{:.0f}_{}_Acreage'.format(y, filedesc)
        fig.tight_layout(pad=3)
        plt.savefig(os.path.join(output_folder, file_name), dpi=300)
        plt.close(fig)
        fig.clf()
        return True

    logging.info('\nCreating Summary Histogram Plots.')
    if plots in ['acreage', 'field']:
        logging.info('Only outputting {} plots.'.format(plots))

    if start_dt and end_dt:
        # custom date range plots
        if plots in ['all', 'acreage']:
            acreage_histogram(ann_grp_df, 'ET_FT', 'Volume_acft',
                              'Total ET: {:%Y-%m-%d} to {:%Y-%m-%d}'
                              .format(start_dt, end_dt),
                              'Total ET (Feet)', 'TotalET')
            acreage_histogram(ann_grp_df, 'NetET_FT', 'NetVolume_acft',
                              'Net ET: {:%Y-%m-%d} to {:%Y-%m-%d}'
                              .format(start_dt, end_dt),
                              'Net ET (Feet)', 'NetET')
        if plots in ['all', 'field']:
            field_count_hist(ann_grp_df, 'ET_FT', 'Volume_acft',
                             'Total ET: {:%Y-%m-%d} to {:%Y-%m-%d}'
                             .format(start_dt, end_dt),
                             'Total ET (Feet)', 'TotalET')
            field_count_hist(ann_grp_df, 'NetET_FT', 'NetVolume_acft',
                             'Net ET: {:%Y-%m-%d} to {:%Y-%m-%d}'
                             .format(start_dt, end_dt),
                             'Net ET (Feet)', 'NetET')
    else:
        # Default Annual and Growing Season Plots if no start/end date
        # Annual Plots
        if plots in ['all', 'acreage']:
            acreage_histogram(ann_grp_df, 'ET_FT', 'Volume_acft',
                              'Annual ET', 'Total ET (Feet)', 'Ann_TotalET')
            acreage_histogram(ann_grp_df, 'NetET_FT', 'NetVolume_acft',
                              'Annual Net ET', 'Net ET (Feet)', 'Ann_NetET')

        if plots in ['all', 'field']:
            field_count_hist(ann_grp_df, 'ET_FT', 'Volume_acft',
                             'Annual ET', 'Total ET (Feet)', 'Ann_TotalET')
            field_count_hist(ann_grp_df, 'NetET_FT', 'NetVolume_acft',
                             'Annual Net ET', 'Net ET (Feet)', 'Ann_NetET')

        # Growing Season Plots
        if 'MONTH' in input_df.columns:
            if plots in ['all', 'acreage']:
                acreage_histogram(gs_grp_df, 'ET_FT', 'Volume_acft',
                                  'Growing Season ET', 'Total ET (Feet)',
                                  'GS_TotalET')
                acreage_histogram(gs_grp_df, 'NetET_FT', 'NetVolume_acft',
                                  'Growing Season Net ET',
                                  'Net ET (Feet)', 'GS_NetET')
            if plots in ['all', 'field']:
                field_count_hist(gs_grp_df, 'ET_FT', 'Volume_acft',
                                 'Growing Season ET', 'Total ET (Feet)',
                                 'GS_TotalET')
                field_count_hist(gs_grp_df, 'NetET_FT', 'NetVolume_acft',
                                 'Growing Season Net ET',
                                 'Net ET (Feet)', 'GS_NetET')


def parse_int_set(nputstr=""):
    """Return list of numbers given a string of ranges
    http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-of-numbers-in-
    python.html"""
    selection = set()
    invalid = set()
    # tokens are comma separated values
    tokens = [x.strip() for x in nputstr.split(',')]
    for i in tokens:
        try:
            # typically tokens are plain old integers
            selection.add(int(i))
        except:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # we have items separated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token) - 1]
                    for x in range(first, last + 1):
                        selection.add(x)
            except:
                # not an int and not a range...
                invalid.add(i)
    # Report invalid tokens before returning valid selection
    # print "Invalid set: " + str(invalid)
    return selection


def valid_date(input_date):
    """Check that a date string is ISO format (YYYY-MM-DD)

    This function is used to check the format of dates entered as command
        line arguments.
    DEADBEEF - It would probably make more sense to have this function
        parse the date using dateutil parser (http://labix.org/python-dateutil)
        and return the ISO format string.

    Parameters
    ----------
    input_date : str

    Returns
    -------
    datetime

    Raises
    ------
    ArgParse ArgumentTypeError

    """
    try:
        return dt.datetime.strptime(input_date, "%Y-%m-%d")
        # return dt.datetime.strptime(input_date, "%Y-%m-%d").date().isoformat()
    except ValueError:
        msg = "Not a valid date: '{}'.".format(input_date)
        raise argparse.ArgumentTypeError(msg)


def arg_parse():
    """"""
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    output_folder = os.path.join(project_folder, 'summary_histograms')

    parser = argparse.ArgumentParser(
        description='Create Histograms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--file', default=None, metavar='FILE PATH',
        help='CSV File Path')
    parser.add_argument(
        '--skip', default=[], type=str, metavar='FID SKIPLIST',
        help='Comma separated list or range of FIDs to skip')
    parser.add_argument(
        '-bmin', default=0, type=int, metavar='HISTOGRAM MIN',
        help='Histogram Minimum (integer)')
    parser.add_argument(
        '-bmax', default=5, type=int, metavar='HISTOGRAM MAX',
        help='Histogram Maximum (integer)')
    parser.add_argument(
        '-bsize', default=0.25, type=int, metavar='HISTOGRAM BIN SIZE',
        help='Histogram Bin Size (integer)')
    parser.add_argument(
        '--start', default=None, type=valid_date,
        metavar='YYYY-MM-DD', help='Start date')
    parser.add_argument(
        '--end', default=None, type=valid_date,
        metavar='YYYY-MM-DD', help='End date')
    parser.add_argument(
        '--plot', default='all', type=str,
        choices=['all', 'acreage', 'field'],
        help='Output Plots: all, acreage, or fields')
    parser.add_argument(
        '--output', default=output_folder, metavar='FOLDER',
        help='Output folder')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    logging.info('{0:<20s} {1}'.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info('{0:<20s} {1}'.format('Current Directory:', os.getcwd()))
    logging.info('{0:<20s} {1}'.format(
        'Script:', os.path.basename(sys.argv[0])))

    main(csv_path=args.file, fid_list=args.skip, bin_min=args.bmin,
         bin_max=args.bmax, bin_size=args.bsize, start_dt=args.start,
         end_dt=args.end, plots=args.plot, output_folder=args.output)
