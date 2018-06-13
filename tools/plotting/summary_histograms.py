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

def main(csv_path, fid_list='', bin_min=0, bin_max=5, bin_size=0.25):

    """Create Summary Histogram Plots from pymetric daily csv output file
    Args:
        csv_path (str): file path of the project INI file
        fid_list (list): list or range of FIDs to skip
        bin_min (int): Histogram Minimum (default: 0)
        bin_max (int): Histogram Max (default: 5)
        bin_size (int): Histogram bin size (default: 0.25)
    Returns:
        None
    """
    logging.info('\nReading Daily csv: {}'.format(csv_path))

    # Check if csv file exist
    if not csv_path:
        logging.error(
            'ERROR: Daily csv file does not exist')
        sys.exit()
    # Attempt to read csv_file
    try:
        daily_df = pd.read_csv(csv_path, sep=',')
    except:
        logging.error('Error reading file. Check csv path.')
        sys.exit()

    def parse_int_set(nputstr=""):
        """Return list of numbers given a string of ranges
        http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-of-numbers-in-
        python.html"""
        selection = set()
        invalid = set()
        # tokens are comma seperated values
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
                        # we have items seperated by a dash
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

    # Filter FIDs based on fid_list (default [])
    fid_skiplist = []
    if fid_list:
        fid_skiplist = sorted(list(parse_int_set(fid_list)))
        print('Skipping FIDs: {}'.format(fid_skiplist))
        daily_df = daily_df[~daily_df['FID'].isin(fid_skiplist)]

    logging.info('\nCreating Summary Histogram Plots')
    # Unit Conversions
    pix2acre = 0.222395 #30x30m pixel to acres; From Google
    mm2ft = 0.00328084 #From Google

    # Add Acres
    daily_df['Area_acres'] = daily_df.PIXELS * pix2acre
    # Add FT Fields
    daily_df['ET_FT'] = daily_df.ET_MM * mm2ft
    daily_df['ETR_FT'] = daily_df.ETR_MM * mm2ft
    # Daily Volume Field
    daily_df['Volume_acft'] = daily_df.Area_acres * daily_df.ET_FT

    # Growing Season Start/End Months (inclusive)
    start_month = 4
    end_month = 10

    # Create Growing Season Only Dataframe
    gs_df = daily_df[(daily_df['MONTH'] >= start_month) &
                     (daily_df['MONTH'] <= end_month)]

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
         'Volume_acft': 'sum'}

    # GS Grouped Dataframe
    gs_grp_df = gs_df.groupby('FID', as_index=True).agg(a)

    # Annual Grouped Dataframe
    ann_grp_df = daily_df.groupby('FID', as_index=True).agg(a)

    # Field Count Histogram Function
    def field_count_hist(grp_df, grp_name):
        # Annotation Box Stats
        y = grp_df['YEAR'].mean()
        m = grp_df['ET_FT'].mean().round(1)
        total_area = grp_df['Area_acres'].sum()
        total_vol = grp_df['Volume_acft'].sum()

        # Bins
        et_bins = np.linspace(bin_min, bin_max,
                              ((bin_max-bin_min) / bin_size) + 1)

        # Make Figure
        font_size = 12
        ann_font_size = 10
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(grp_df.ET_FT, bins=et_bins, align='mid', edgecolor='black')
        ax.set_title(grp_name, size=font_size)
        ax.set_xlabel('Total ET (FT)', size=font_size)
        ax.set_ylabel('Field Count', size=font_size)
        ax.set_xticks(np.arange(0, bin_max+ (2*bin_size), 2*bin_size))
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        ymin, ymax = plt.ylim()  # return the current ylim
        plt.ylim((ymin, ymax + ymax * 0.3))  # shift ymax for annotation space
        # Add mean vertical line
        ax.axvline(m, color='gray', linestyle='dashed',
                   linewidth=1)

        # Add Annotation Text Box
        an1Text = (
                'Year {:.0f}\n' +
                'Mean ET = {:.1f} ft\n' +
                'Total Area = {:,.1f} acres\n' +
                'Total ET Volume = {:,.1f} ac-ft').format(
            y, m, total_area, total_vol)
        at = AnchoredText(
            an1Text, prop=dict(size=ann_font_size), frameon=True, loc=2)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        #Save Figure
        file_name = '{:.0f}_{}_Fields'.format(y, grp_name.replace(" ", "_"))
        fig.tight_layout(pad=3)
        plt.savefig(file_name, dpi=300)
        plt.close(fig)
        fig.clf()
        return True

    # Acreage histogram (Bar Plot)
    def acreage_histogram(grp_df, grp_name):
        # Annotation Box Stats
        y = grp_df['YEAR'].mean()
        m = grp_df['ET_FT'].mean().round(1)
        total_area = grp_df['Area_acres'].sum()
        total_vol = grp_df['Volume_acft'].sum()

        # Bins
        et_bins = np.linspace(bin_min, bin_max,
                              ((bin_max - bin_min) / bin_size)+ 1)

        # Acreage/ET Bins
        et_area_hist, et_bins, binnum = stats.binned_statistic(
            grp_df.ET_FT, grp_df.Area_acres, 'sum', et_bins)

        # Make Figure
        font_size = 12
        ann_font_size=10
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(et_bins[:-1], et_area_hist, width=bin_size, edgecolor='black',
               align='edge', color='r')
        ax.set_title(grp_name, size=font_size)
        ax.set_xlabel('Total ET (FT)', size=font_size)
        ax.set_ylabel('Acreage', size=font_size)
        ax.set_xticks(np.arange(0, bin_max+ (2*bin_size), 2*bin_size))
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        ymin, ymax = plt.ylim()  # return the current ylim
        plt.ylim((ymin, ymax + ymax*0.3))  # shift ymax for annotation space
        # Add mean vertical line
        ax.axvline(m, color='gray', linestyle='dashed',
                   linewidth=1)

        # Add Annotation Text Box
        an1Text = (
                'Year {:.0f}\n' +
                'Mean ET = {:.1f} ft\n' +
                'Total Area = {:.1f} acres\n' +
                'Total ET Volume = {:.1f} ac-ft').format(
            y, m, total_area, total_vol)
        at = AnchoredText(
            an1Text, prop=dict(size=ann_font_size), frameon=True, loc=2)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        #Save Figure
        file_name = '{:.0f}_{}_Acreage'.format(y, grp_name.replace(" ", "_"))
        fig.tight_layout(pad=3)
        plt.savefig(file_name, dpi=300)
        plt.close(fig)
        fig.clf()
        return True

    # Field-count Plots
    field_count_hist(gs_grp_df, 'Growing Season ET')
    field_count_hist(ann_grp_df, 'Annual ET')

    # Acreage-based Plots
    acreage_histogram(gs_grp_df, 'Growing Season ET')
    acreage_histogram(ann_grp_df, 'Annual ET')

def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Create Histograms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-f', '--file', default=None, metavar='FILE PATH',
        help='CSV File Path')
    parser.add_argument(
        '-s', '--skip', default=[], type=str, metavar='FID SKIPLIST',
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
    # parser.add_argument(
    #     '-o', '--overwrite', default=False, action="store_true",
    #     help='Force overwrite of existing files')
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
         bin_max=args.bmax, bin_size=args.bsize)
