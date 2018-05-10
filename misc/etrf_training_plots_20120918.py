# Name:         etrf_training_plots_20120918.py
# Purpose:      Plots for thesis
# Author:       Charles Morton
# Created       2012/09/18
# Copyright:    (c) DRI
# ArcGIS:       10 SP2
# Python:       2.6
#--------------------------------

import calendar
from datetime import datetime
import logging
import math
import os
import re

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
##from scipy import *
from scipy import optimize
from scipy.stats import cumfreq

################################################################################

def etrf_training_plots(workspace):
    try:
        show_flag = True
        save_flag = True

        workspace = os.getcwd()
        plots_ws = "%s\\PLOTS" % workspace
        if not os.path.isdir(plots_ws): os.makedirs(plots_ws)

        ##data = np.recfromcsv(r"P:\supportdata\personal\morton\THESIS\etrf_training.csv")
        ##data = np.recfromcsv(r"P:\supportdata\personal\morton\THESIS\etrf_training_jawra.csv")
        data = np.recfromcsv(r"D:\NASA\MISC\etrf_training.csv")

        def prob_array(data_array):
            array_len = len(data_array)
            return np.linspace(0., array_len, array_len)/array_len


        ##data = data[data['pixelcount']>=10000]
        ##data = data[data['dt']>=4]
        ##data = data[data['doy']>=90]
        ##data = data[data['doy']<=300]
        #### This removes some bad calibrations:
        #### 2009/LG/43/216, 2006/AS/42/137, 2006/AS/42/153
        #### 2009/AS/42/289 has very little temp different (~7)
        ##data = data[data['kc_cld_pct']<=12]
        ##data = data[data['kc_hot_pct']<=16]

        all_cold_data = np.insert(np.sort(data['kc_cld_pct']), 0, 0)
        all_hot_data = np.insert(np.sort(data['kc_hot_pct']), 0, 0)

        plt.figure()
        plt.plot(all_cold_data, prob_array(all_cold_data), 'b-', label='Cold')
        plt.plot(all_hot_data, prob_array(all_hot_data), 'r-', label='Hot')
        plt.ylim(0,1)
        plt.title('Cold/Hot Calibration - All Data')
        plt.ylabel('Probability')
        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
        plt.legend(loc=4)
        if save_flag:
            plt.savefig('%s\\all_data_cdf.png' % plots_ws, bbox_inches='tight')
        if show_flag: plt.show()
        plt.close()





##        #### CDF by area
##        fm_data = np.insert(np.sort(data[data['area']=='FALLON/MASON']['kc_cld_pct']), 0, 0)
##        ##cm_data = np.insert(np.sort(data[data['area']=='CARSON/MASON']['kc_cld_pct']), 0, 0)
##        ms_data = np.insert(np.sort(data[data['area']=='MASON/SMITH']['kc_cld_pct']), 0, 0)
##        plt.figure()
##        plt.plot(all_cold_data, prob_array(all_cold_data), 'k-', label='All')
##        plt.plot(fm_data, prob_array(fm_data), 'b-', label='Fallon/Mason')
##        ##plt.plot(cm_data, prob_array(cm_data), 'g-', label='Carson/Mason')
##        plt.plot(ms_data, prob_array(ms_data), 'r-', label='Mason/Smith')
##        plt.legend(loc=4)
##        plt.ylim(0,1)
##        plt.title('Cold Calibration by Area - All Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\all_data_cold_cdf_area.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##        fm_data = np.insert(np.sort(data[data['area']=='FALLON/MASON']['kc_hot_pct']), 0, 0)
##        ##cm_data = np.insert(np.sort(data[data['area']=='CARSON/MASON']['kc_hot_pct']), 0, 0)
##        ms_data = np.insert(np.sort(data[data['area']=='MASON/SMITH']['kc_hot_pct']), 0, 0)
##        plt.figure()
##        plt.plot(all_hot_data, prob_array(all_hot_data), 'k-', label='All')
##        plt.plot(fm_data, prob_array(fm_data), 'b-', label='Fallon/Mason')
##        ##plt.plot(cm_data, prob_array(cm_data), 'g-', label='Carson/Mason')
##        plt.plot(ms_data, prob_array(ms_data), 'r-', label='Mason/Smith')
##        plt.legend(loc=4)
##        plt.ylim(0,1)
##        plt.title('Hot Calibration by Area - All Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\all_data_hot_cdf_area.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### CDF by user
##        user_list = list(set(data['user']))
##        plt.figure()
##        plt.plot(all_cold_data, prob_array(all_cold_data), 'k-', label='All')
##        for i, user in enumerate(user_list):
##            user_cld_data = np.insert(np.sort(data[data['user']==user]['kc_cld_pct']), 0, 0)
##            plt.plot(user_cld_data, prob_array(user_cld_data),
##                     '-', label=user, c=cm.hsv(1.*i/len(user_list)))
##        plt.legend(loc=4)
##        plt.ylim(0,1)
##        plt.title('Cold Calibration by User - All Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\all_data_cold_cdf_user.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##        plt.figure()
##        plt.plot(all_hot_data, prob_array(all_hot_data), 'k-', label='All')
##        for i, user in enumerate(user_list):
##            user_hot_data = np.insert(np.sort(data[data['user']==user]['kc_hot_pct']), 0, 0)
##            plt.plot(user_hot_data, prob_array(user_hot_data),
##                     '-', label=user, c=cm.hsv(1.*i/len(user_list)))
##        plt.legend(loc=4)
##        plt.ylim(0,1)
##        plt.title('Hot Calibration by User - All Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\all_data_hot_cdf_user.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        ###### CDF by year
##        ##year_list = list(set(data['year']))
##        ##plt.figure()
##        ##plt.plot(all_cold_data, prob_array(all_cold_data), 'k-', label='All')
##        ##for i, year in enumerate(year_list):
##        ##    year_cld_data = np.insert(np.sort(data[data['year']==year]['kc_cld_pct']), 0, 0)
##        ##    plt.plot(year_cld_data, prob_array(year_cld_data),
##        ##             '-', label=str(year), c=cm.hsv(1.*i/len(year_list)))
##        ##plt.legend(loc=4)
##        ##plt.ylim(0,1)
##        ##plt.title('Cold Calibration by Year - All Data')
##        ##plt.ylabel('Probability')
##        ##plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        ##if save_flag:
##        ##    plt.savefig('%s\\all_data_cold_cdf_year.png' % plots_ws, bbox_inches='tight')
##        ##if show_flag: plt.show()
##        ##plt.close()
##        ##plt.figure()
##        ##plt.plot(all_hot_data, prob_array(all_hot_data), 'k-', label='All')
##        ##for i, year in enumerate(year_list):
##        ##    year_hot_data = np.insert(np.sort(data[data['year']==year]['kc_hot_pct']), 0, 0)
##        ##    plt.plot(year_hot_data, prob_array(year_hot_data),
##        ##             '-', label=str(year), c=cm.hsv(1.*i/len(year_list)))
##        ##plt.legend(loc=4)
##        ##plt.ylim(0,1)
##        ##plt.title('Hot Calibration by Year - All Data')
##        ##plt.ylabel('Probability')
##        ##plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        ##if save_flag:
##        ##    plt.savefig('%s\\all_data_hot_cdf_year.png' % plots_ws, bbox_inches='tight')
##        ##if show_flag: plt.show()
##        ##plt.close()
##
##        #### Tail size vs DOY
##        plt.figure()
##        plt.plot(data['doy'], data['kc_cld_pct'], 'b.', label='Cold')
##        ##user_list = list(set(data['user']))
##        ##for i, user in enumerate(user_list):
##        ##    plt.plot(data[data['user']==user]['doy'], data[data['user']==user]['kc_cld_pct'],
##        ##             '.', label=user, c=cm.hsv(1.*i/len(user_list)))
##        plt.xlim(1,365)
##        plt.title('All Cold Calibrations')
##        plt.xlabel('Day of Year')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        ##plt.legend(loc=1)
##        if save_flag:
##            plt.savefig('%s\\all_data_cold_etrf_vs_doy.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##        plt.figure()
##        plt.plot(data['doy'], data['kc_hot_pct'], 'r.', label='Hot')
##        ##user_list = list(set(data['user']))
##        ##for i, user in enumerate(user_list):
##        ##    plt.plot(data[data['user']==user]['doy'], data[data['user']==user]['kc_hot_pct'],
##        ##             '.', label=user, c=cm.hsv(1.*i/len(user_list)))
##        plt.xlim(1,365)
##        plt.title('All Hot Calibrations')
##        plt.xlabel('Day of Year')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        ##plt.legend(loc=1)
##        if save_flag:
##            plt.savefig('%s\\all_data_hot_etrf_vs_doy.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()

##        #### Tail size vs DOY
##        plt.figure()
##        print "ALL_COLD:", len(data['kc_cld_pct'])
##        print "ALL_HOT: ", len(data['kc_hot_pct'])
##        plt.plot(data['doy'], data['kc_hot_pct'], 'o',
##                 label='Hot Tail Size', ms=4, mew=0, mfc='r')
##        plt.plot(data['doy'], data['kc_cld_pct'], '^',
##                 label='Cold Tail Size', ms=4, mew=0, mfc='b')
##        plt.xlim(1,365)
##        ##plt.title('All Calibrations')
##        plt.xlabel('Day of Year')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        plt.legend(loc=1, numpoints=1)
##        if save_flag:
##            plt.savefig('%s\\all_data_etrf_vs_doy.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()

##        #### Tail size vs PixelCount
##        plt.figure()
##        plt.plot(data['pixelcount'], data['kc_cld_pct'], 'b.', label='Cold')
##        plt.title('Cold Calibration - All Data')
##        plt.xlabel('PixelCount')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\all_data_cold_etrf_vs_pixelcount.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##        plt.figure()
##        plt.plot(data['pixelcount'], data['kc_hot_pct'], 'r.', label='Hot')
##        plt.title('Hot Calibration - All Data')
##        plt.xlabel('Day of Year')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\all_data_hot_etrf_vs_pixelcount.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### Tail size vs dT
##        plt.figure()
##        plt.plot(data['dt'], data['kc_cld_pct'], 'b.', label='Cold')
##        plt.title('All Cold Calibrations')
##        plt.xlabel('dT (Hot Ts - Cold Ts)')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\all_data_cold_etrf_vs_dt.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##        plt.figure()
##        plt.plot(data['dt'], data['kc_hot_pct'], 'r.', label='Hot')
##        plt.title('All Hot Calibrations')
##        plt.xlabel('dT (Hot Ts - Cold Ts)')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\all_data_hot_etrf_vs_dt.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### Tail size vs dT
##        plt.figure()
##        plt.plot(data['dt'], data['kc_hot_pct'], 'o',
##                 label='Hot Tail Size', ms=4, mew=0, mfc='r')
##        plt.plot(data['dt'], data['kc_cld_pct'], '^',
##                 label='Cold Tail Size', ms=4, mew=0, mfc='b')
####        plt.title('All Calibrations')
##        plt.xlabel('dT (Hot Ts - Cold Ts)')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        plt.legend(loc=1, numpoints=1)
##        if save_flag:
##            plt.savefig('%s\\all_data_etrf_vs_dt.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### dT vs DOY
##        plt.figure()
##        plt.plot(data['doy'], data['dt'], 'k.')
##        plt.xlim(0,366)
##        plt.title('dT (Hot Ts - Cold Ts)')
##        plt.xlabel('Day of Year')
##        plt.ylabel('dT (Ts_Hot-Ts_Cold)')
##        if save_flag:
##            plt.savefig('%s\\all_data_dt_vs_doy.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### PixelCount vs DOY
##        plt.figure()
##        plt.plot(data['doy'], data['pixelcount'], 'k.')
##        plt.xlim(0,366)
##        plt.title('PixelCount vs DOY - All Data')
##        plt.xlabel('Day of Year')
##        plt.ylabel('PixelCount')
##        if save_flag:
##            plt.savefig('%s\\all_data_pixelcount_vs_doy.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()





##        #### Separate by DOY range
##        import matplotlib.cm as cm
##        ##doy_range = [(1,90),(91,151),(151,200),(201,250),(251,300),(301,366)]
##        ##doy_range = [(1,60),(61,151),(201,300),(301,366)]
##        ##doy_range = [(1,60),(61,151),(152,243),(244,366)]
##        ##doy_range = [
##        ##    (1,50),(51,100),(101,150),(151,200),(201,250),(251,380),(300,366)]
##        ##doy_range = [
##        ##    (1,31),(32,59),(60,90),(91,120),(121,151),(152,181),(182,212),
##        ##    (213,243),(244,273),(274,304),(305,334),(335,366)]
##        ##doy_range = [
##        ##    (1,30),(31,60),(61,90),(91,120),(121,150),(151,180),(181,210),
##        ##    (211,240),(241,270),(271,300),(301,330),(331,366)]
##        doy_range = [
##            (1,30),(31,60),(61,90),(91,120),(121,150),(151,180),(181,210),
##            (211,240),(241,270),(271,300),(301,366)]
##        data_boxplot_cold = []
##        data_boxplot_hot = []
##        doy_range_label_cold = []
##        doy_range_label_hot = []
##        for i, (doy_min, doy_max) in enumerate(doy_range):
##            data_boxplot_mask = np.logical_and(
##                data['doy']>=doy_min, data['doy']<=doy_max)
##            if not np.any(data_boxplot_mask): continue
##            ##doy_range_label.append("{0}-{1}".format(doy_min,doy_max))
##            ##doy_range_label.append(calendar.month_abbr[i+1])
##            data_boxplot_cold.append(data[data_boxplot_mask]['kc_cld_pct'])
##            data_boxplot_hot.append(data[data_boxplot_mask]['kc_hot_pct'])
##            if i==10:
##                #### Adjust November month label to be Nov/Dec
##                month_str = "Nov/Dec"
##            else:
##                month_str = calendar.month_abbr[i+1]
##            doy_range_label_cold.append("{0}\n({1})".format(
##                month_str, len(data[data_boxplot_mask]['kc_cld_pct'])))
##            doy_range_label_hot.append("{0}\n({1})".format(
##                month_str, len(data[data_boxplot_mask]['kc_hot_pct'])))
##            print i, doy_min, doy_max, len(data[data_boxplot_mask]['kc_cld_pct'])
##            del data_boxplot_mask
##
##        
##        fig = plt.figure()
##        ax = fig.add_subplot(111)
##        bp = ax.boxplot(data_boxplot_cold)
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        xtickNames = plt.setp(ax, xticklabels=doy_range_label_cold)
##        plt.setp(xtickNames, rotation=0, fontsize=12)
##        ##ax1.set_xlabel('DOY Ranges')
##        ##plt.xlabel('Month')
##        plt.xlabel('Month\n(number of calibrations)')
##        plt.text(0.9, 0.88, 'a', size=20, transform = ax.transAxes)
##        plt.setp(bp['medians'], color='k')
##        plt.setp(bp['boxes'], color='k')
##        plt.setp(bp['whiskers'], color='k',  linestyle='-' )
##        plt.setp(bp['fliers'], color='k')
##        if save_flag:
##            plt.savefig('{0}\\all_data_cold_vs_doy_boxplot.png'.format(plots_ws),
##                        bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##        
##        fig = plt.figure()
##        ax = fig.add_subplot(111)
##        bp = ax.boxplot(data_boxplot_hot)
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        xtickNames = plt.setp(ax, xticklabels=doy_range_label_hot)
##        plt.setp(xtickNames, rotation=0, fontsize=12)
##        ##ax1.set_xlabel('DOY Ranges')
##        ##plt.xlabel('Month')
##        plt.xlabel('Month\n(number of calibrations)')
##        plt.text(0.9, 0.88, 'b', size=20, transform = ax.transAxes)
##        plt.setp(bp['medians'], color='k')
##        plt.setp(bp['boxes'], color='k')
##        plt.setp(bp['whiskers'], color='k',  linestyle='-' )
##        plt.setp(bp['fliers'], color='k')
##        if save_flag:
##            plt.savefig('{0}\\all_data_hot_vs_doy_boxplot.png'.format(plots_ws),
##                        bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()







##        #### Apply pixel_count filter and remove CARSON/MASON
##        data_sub = data[np.logical_or(data['area']=='FALLON/MASON',data['area']=='MASON/SMITH')]
##        ##
##        ##data_sub = data_sub[~np.logical_or(data['user']=='AS',data['user']=='JB')]
##
##        print data_sub.shape
##        print len(data_sub)
##
##        data_sub = data_sub[data_sub['year']<>2006]
##        data_sub = data_sub[data_sub['pixelcount']>=10000]
##        data_sub = data_sub[data_sub['dt']>=4]
##        data_sub = data_sub[data_sub['doy']>=90]
##        data_sub = data_sub[data_sub['doy']<=300]
##        #### This removes some bad calibrations:
##        #### 2009/LG/43/216, 2006/AS/42/137, 2006/AS/42/153
##        #### 2009/AS/42/289 has very little temp different (~7)
##        data_sub = data_sub[data_sub['kc_cld_pct']<=12]
##        data_sub = data_sub[data_sub['kc_hot_pct']<=16]
##        ##print data_sub[data_sub['kc_cld_pct']>16]
##        ##print data_sub[data_sub['kc_hot_pct']>16]
##
##        print data_sub.shape
##        print len(data_sub)
##
##        sub_cold_data = np.insert(np.sort(data_sub['kc_cld_pct']), 0, 0)
##        sub_hot_data = np.insert(np.sort(data_sub['kc_hot_pct']), 0, 0)
##
##        plt.figure()
##        plt.plot(sub_cold_data, prob_array(sub_cold_data), 'k--',
##                 label='Cold Threshold Tail Sizes', linewidth=2)
##        plt.plot(sub_hot_data, prob_array(sub_hot_data), 'k-',
##                 label='Hot Threshold Tail Sizes', linewidth=2)
##        plt.ylim(0,1)
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        plt.legend(loc=4)
##        if save_flag:
##            plt.savefig('%s\\sub_data_cdf.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()

##        plt.figure()
##        plt.plot(sub_cold_data, prob_array(sub_cold_data), 'b-', label='Cold')
##        plt.plot(sub_hot_data, prob_array(sub_hot_data), 'r-', label='Hot')
##        plt.ylim(0,1)
##        plt.title('Cold/Hot Calibration - Sub Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        plt.legend(loc=4)
##        if save_flag:
##            plt.savefig('%s\\sub_data_cdf.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()

##        #### CDF by area
##        fm_data = np.insert(np.sort(data_sub[data_sub['area']=='FALLON/MASON']['kc_cld_pct']), 0, 0)
##        ms_data = np.insert(np.sort(data_sub[data_sub['area']=='MASON/SMITH']['kc_cld_pct']), 0, 0)
##        plt.figure()
##        plt.plot(sub_cold_data, prob_array(sub_cold_data), 'k-', label='All')
##        plt.plot(fm_data, prob_array(fm_data), 'b-', label='Fallon/Mason')
##        plt.plot(ms_data, prob_array(ms_data), 'r-', label='Mason/Smith')
##        plt.legend(loc=4)
##        plt.ylim(0,1)
##        plt.title('Cold Calibration by Area - Sub Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\sub_data_cold_cdf_area.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##        fm_data = np.insert(np.sort(data_sub[data_sub['area']=='FALLON/MASON']['kc_hot_pct']), 0, 0)
##        ms_data = np.insert(np.sort(data_sub[data_sub['area']=='MASON/SMITH']['kc_hot_pct']), 0, 0)
##        plt.figure()
##        plt.plot(sub_hot_data, prob_array(sub_hot_data), 'k-', label='All')
##        plt.plot(fm_data, prob_array(fm_data), 'b-', label='Fallon/Mason')
##        plt.plot(ms_data, prob_array(ms_data), 'r-', label='Mason/Smith')
##        plt.legend(loc=4)
##        plt.ylim(0,1)
##        plt.title('Hot Calibration by Area - Sub Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\sub_data_hot_cdf_area.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### CDF by user
##        user_list = list(set(data['user']))
##        plt.figure()
##        plt.plot(sub_cold_data, prob_array(sub_cold_data), 'k-', label='All')
##        for i, user in enumerate(user_list):
##            user_cld_data = np.insert(np.sort(data_sub[data_sub['user']==user]['kc_cld_pct']), 0, 0)
##            plt.plot(user_cld_data, prob_array(user_cld_data),
##                     '-', label=user, c=cm.hsv(1.*i/len(user_list)))
##        plt.legend(loc=4)
##        plt.ylim(0,1)
##        plt.title('Cold Calibration by User - Sub Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\sub_data_cold_cdf_user.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##        plt.figure()
##        plt.plot(sub_hot_data, prob_array(sub_hot_data), 'k-', label='All')
##        for i, user in enumerate(user_list):
##            user_hot_data = np.insert(np.sort(data_sub[data_sub['user']==user]['kc_hot_pct']), 0, 0)
##            plt.plot(user_hot_data, prob_array(user_hot_data),
##                     '-', label=user, c=cm.hsv(1.*i/len(user_list)))
##        plt.legend(loc=4)
##        plt.ylim(0,1)
##        plt.title('Hot Calibration by User - Sub Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\sub_data_hot_cdf_user.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### CDF by year
##        year_list = list(set(data['year']))
##        plt.figure()
##        plt.plot(sub_cold_data, prob_array(sub_cold_data), 'k-', label='All')
##        for i, year in enumerate(year_list):
##            year_cld_data = np.insert(np.sort(data_sub[data_sub['year']==year]['kc_cld_pct']), 0, 0)
##            plt.plot(year_cld_data, prob_array(year_cld_data),
##                     '-', label=str(year), c=cm.hsv(1.*i/len(year_list)))
##        plt.legend(loc=4)
##        plt.ylim(0,1)
##        plt.title('Cold Calibration by Year - Sub Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\sub_data_cold_cdf_year.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##        plt.figure()
##        plt.plot(sub_hot_data, prob_array(sub_hot_data), 'k-', label='All')
##        for i, year in enumerate(year_list):
##            year_hot_data = np.insert(np.sort(data_sub[data_sub['year']==year]['kc_hot_pct']), 0, 0)
##            plt.plot(year_hot_data, prob_array(year_hot_data),
##                     '-', label=str(year), c=cm.hsv(1.*i/len(year_list)))
##        plt.legend(loc=4)
##        plt.ylim(0,1)
##        plt.title('Hot Calibration by Year - Sub Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\sub_data_hot_cdf_year.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### CDF by DOY range
##        doy_ranges = [(91,150),(151,200),(201,250),(251,300)]
##        ##doy_ranges = [(1,60),(61,151),(201,300),(301,366)]
##        ##doy_ranges = [(1,60),(61,151),(152,243),(244,366)]
##        ##doy_ranges = [(61,90),(91,120),(121,150),(151,180),(181,210),
##        ##              (211,240),(241,270),(271,300),(301,330),(331,366)]
##        plt.figure()
##        plt.plot(sub_cold_data, prob_array(sub_cold_data), 'k-', label='All')
##        for i, (doy_min, doy_max) in enumerate(doy_ranges):
##            doy_cld_data = np.insert(np.sort(data_sub[
##                (data_sub['doy']>=doy_min)&(data_sub['doy']<=doy_max)]['kc_cld_pct']), 0, 0)
##            plt.plot(doy_cld_data, prob_array(doy_cld_data),
##                     '-', label=("%s-%s"%(doy_min,doy_max)),
##                     c=cm.hsv(1.*i/len(doy_ranges)))
##        plt.legend(loc=4)
##        plt.ylim(0,1)
##        ##plt.title('Cold Calibration by DOY range - Sub Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\sub_data_cold_cdf_doy.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##        plt.figure()
##        plt.plot(sub_hot_data, prob_array(sub_hot_data), 'k-', label='All')
##        for i, (doy_min, doy_max) in enumerate(doy_ranges):
##            doy_hot_data = np.insert(np.sort(data_sub[
##                (data_sub['doy']>=doy_min)&(data_sub['doy']<=doy_max)]['kc_hot_pct']), 0, 0)
##            plt.plot(doy_hot_data, prob_array(doy_hot_data),
##                     '-', label=("%s-%s"%(doy_min,doy_max)),
##                     c=cm.hsv(1.*i/len(doy_ranges)))
##        plt.legend(loc=4)
##        plt.ylim(0,1)
##        ##plt.title('Hot Calibration by DOY range - Sub Data')
##        plt.ylabel('Probability')
##        plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\sub_data_hot_cdf_doy.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### Tail size vs DOY
##        plt.figure()
##        plt.plot(data_sub['doy'], data_sub['kc_cld_pct'], 'b.', label='Cold')
##        ##user_list = list(set(data_sub['user']))
##        ##for i, user in enumerate(user_list):
##        ##    plt.plot(data_sub[data_sub['user']==user]['doy'],
##        ##             data_sub[data_sub['user']==user]['kc_cld_pct'],
##        ##             '.', label=user, c=cm.hsv(1.*i/len(user_list)))
##        plt.xlim(1,365)
##        plt.title('Filtered Cold Calibrations')
##        plt.xlabel('Day of Year')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        ##plt.legend(loc=1)
##        if save_flag:
##            plt.savefig('%s\\sub_data_cold_etrf_vs_doy.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##        plt.figure()
##        plt.plot(data_sub['doy'], data_sub['kc_hot_pct'], 'r.', label='Hot')
##        ##user_list = list(set(data_sub['user']))
##        ##for i, user in enumerate(user_list):
##        ##    plt.plot(data_sub[data_sub['user']==user]['doy'],
##        ##             data_sub[data_sub['user']==user]['kc_hot_pct'],
##        ##             '.', label=user, c=cm.hsv(1.*i/len(user_list)))
##        plt.xlim(1,365)
##        plt.title('Filtered Hot Calibrations')
##        plt.xlabel('Day of Year')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        ##plt.legend(loc=1)
##        if save_flag:
##            plt.savefig('%s\\sub_data_hot_etrf_vs_doy.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()

##        #### Tail size vs DOY
##        plt.figure()
##        print "SUB_COLD:", len(data_sub['kc_cld_pct'])
##        print "SUB_HOT: ", len(data_sub['kc_hot_pct'])
##        plt.plot(data_sub['doy'], data_sub['kc_hot_pct'], 'o',
##                 label='Hot Tail Size', ms=4, mew=0, mfc='r')
##        plt.plot(data_sub['doy'], data_sub['kc_cld_pct'], '^',
##                 label='Cold Tail Size', ms=4, mew=0, mfc='b')
##        plt.xlim(1,365)
##        ##plt.title('Filtered Calibrations')
##        plt.xlabel('Day of Year')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        plt.legend(loc=1, numpoints=1)
##        if save_flag:
##            plt.savefig('%s\\sub_data_etrf_vs_doy.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()

##        #### Tail size vs PixelCount
##        plt.figure()
##        plt.plot(data_sub['pixelcount'], data_sub['kc_cld_pct'], 'b.', label='Cold')
##        plt.title('Cold Calibration - Sub Data')
##        plt.xlabel('PixelCount')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\sub_data_cold_etrf_vs_pixelcount.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##        plt.figure()
##        plt.plot(data_sub['pixelcount'], data_sub['kc_hot_pct'], 'r.', label='Hot')
##        plt.title('Hot Calibration - Sub Data')
##        plt.xlabel('Day of Year')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\sub_data_hot_etrf_vs_pixelcount.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### Tail size vs dT
##        plt.figure()
##        plt.plot(data_sub['dt'], data_sub['kc_cld_pct'], 'b.', label='Cold')
##        plt.title('Filtered Cold Calibrations')
##        plt.xlabel('dT (Hot Ts - Cold Ts)')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag: plt.savefig('%s\\sub_data_cold_etrf_vs_dt.png' % plots_ws)
##        if show_flag: plt.show()
##        plt.close()
##        plt.figure()
##        plt.plot(data_sub['dt'], data_sub['kc_hot_pct'], 'r.', label='Hot')
##        plt.title('Filtered Hot Calibrations')
##        plt.xlabel('dT (Hot Ts - Cold Ts)')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        if save_flag:
##            plt.savefig('%s\\sub_data_hot_etrf_vs_dt.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### Tail size vs dT
##        plt.figure()
##        plt.plot(data_sub['dt'], data_sub['kc_hot_pct'], 'o',
##                 label='Hot Tail Size', ms=4, mew=0, mfc='r')
##        plt.plot(data_sub['dt'], data_sub['kc_cld_pct'], '^',
##                 label='Cold Tail Size', ms=4, mew=0, mfc='b')
####        plt.title('Filtered Calibrations')
##        plt.xlabel('dT (Hot Ts - Cold Ts)')
##        plt.ylabel('Percentage of agricultural pixels outside calibration threshold')
##        plt.legend(loc=1, numpoints=1)
##        if save_flag:
##            plt.savefig('%s\\sub_data_etrf_vs_dt.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### dT vs DOY
##        plt.figure()
##        plt.plot(data_sub['doy'], data_sub['dt'], 'k.')
##        plt.xlim(0,366)
##        plt.title('dT vs DOY - Sub Data')
##        plt.xlabel('Day of Year')
##        plt.ylabel('dT (Ts_Hot-Ts_Cold)')
##        if save_flag:
##            plt.savefig('%s\\sub_data_dt_vs_doy.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()
##
##        #### PixelCount vs DOY
##        plt.figure()
##        plt.plot(data_sub['doy'], data_sub['pixelcount'], 'k.')
##        plt.xlim(0,366)
##        plt.title('PixelCount vs DOY - Sub Data')
##        plt.xlabel('Day of Year')
##        plt.ylabel('PixelCount')
##        if save_flag:
##            plt.savefig('%s\\sub_data_pixelcount_vs_doy.png' % plots_ws, bbox_inches='tight')
##        if show_flag: plt.show()
##        plt.close()


        ###### Separate by DOY range
        ##import matplotlib.cm as cm
        ######doy_ranges = [(1,60),(61,151),(201,300),(301,366)]
        ####doy_ranges = [(1,60),(61,151),(152,243),(244,366)]
        ##doy_ranges = [(61,90),(91,120),(121,150),(151,180),(181,210),
        ##              (211,240),(241,270),(271,300),(301,330),(331,366)]
        ##n = len(doy_ranges)
        ##
        ##plt.figure()
        ##plt.plot(all_cold_data, prob_array(all_cold_data), 'k-', label='All')
        ##for i, (doy_min, doy_max) in enumerate(doy_ranges):
        ##    data_sub = data[np.logical_and(data['doy']>=doy_min, data['doy']<=doy_max)]
        ##    user_data = np.insert(np.sort(data_sub['kc_cld_pct']), 0, 0)
        ##    plt.plot(user_data, prob_array(user_data), c=cm.hsv(1.*i/n), 
        ##             label=("%d - %d" % (doy_min, doy_max)))
        ##plt.legend(loc=4)
        ##plt.ylim(0,1)
        ##plt.title('Cold Calibration by DOY Range')
        ##plt.ylabel('Probability')
        ##plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
        ##if save_flag: plt.savefig('%s\\cold_cdf_doy.png' % plots_ws)
        ##if show_flag: plt.show()
        ##plt.close()
        ##
        ##plt.figure()
        ##plt.plot(all_hot_data, prob_array(all_hot_data), 'k-', label='All')
        ##for i, (doy_min, doy_max) in enumerate(doy_ranges):
        ##    data_sub = data[np.logical_and(data['doy']>=doy_min, data['doy']<=doy_max)]
        ##    user_data = np.insert(np.sort(data_sub['kc_cld_pct']), 0, 0)
        ##    plt.plot(user_data, prob_array(user_data), c=cm.hsv(1.*i/n), 
        ##             label=("%d - %d" % (doy_min, doy_max)))
        ##plt.legend(loc=4)
        ##plt.ylim(0,1)
        ##plt.title('Hot Calibration by DOY Range')
        ##plt.ylabel('Probability')
        ##plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
        ##if save_flag: plt.savefig('%s\\hot_cdf_doy.png' % plots_ws)
        ##if show_flag: plt.show()
        ##plt.close()



        ###### Compare CDF of original data and subdata
        ##plt.figure()
        ##plt.plot(sub_cold_data, prob_array(sub_cold_data), 'b-', linewidth='1.5',
        ##         label='Subset Cold Tail Sizes')
        ##plt.plot(sub_hot_data, prob_array(sub_hot_data), 'r-', linewidth='1.5',
        ##         label='Subset Hot Tail Sizes')
        ##plt.plot(all_cold_data, prob_array(all_cold_data), 'b--', linewidth='1.5',
        ##         label='All Cold Tail Sizes')
        ##plt.plot(all_hot_data, prob_array(all_hot_data), 'r--', linewidth='1.5',
        ##         label='All Hot Tail Sizes')
        ##plt.ylim(0,1)
        ##plt.xlim(0,20)
        ####plt.title('Comparison of Original and Filtered User Calibrations')
        ##plt.ylabel('Probability')
        ##plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
        ##plt.legend(loc=4)
        ##if save_flag:
        ##    plt.savefig('%s\\all_sub_comparison_cdf.png' % plots_ws, bbox_inches='tight')
        ##if show_flag: plt.show()
        ##plt.close()



        ###### Filters
        ##data = data[data['doy']>=70]
        ##data = data[data['doy']<=300]
        ####data = data[np.logical_or(data['area']=='FALLON/MASON',data['area']=='CARSON/MASON')]
        ##data = data[data['pixelcount']>=10000]
        ##
        ###### Remove bad calibrations
        #### 2006 path_42 153 AS
        ##
        ###### All data
        ##data_sub = data[np.logical_or(data['area']=='FALLON/MASON',data['area']=='CARSON/MASON')]
        ##all_cold_data = np.insert(np.sort(data_sub['kc_cld_pct']), 0, 0)
        ##all_hot_data = np.insert(np.sort(data_sub['kc_hot_pct']), 0, 0)
        ##
        ##plt.figure()
        ##plt.plot(all_cold_data, prob_array(all_cold_data), 'b-', label='Cold')
        ##plt.plot(all_hot_data, prob_array(all_hot_data), 'r-', label='Hot')
        ##plt.ylim(0,1)
        ##plt.title('Cold/Hot Calibration - DOY 70 to DOY 300')
        ##plt.ylabel('Probability')
        ##plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
        ##plt.legend(loc=4)
        ##if save_flag: plt.savefig('%s\\cold_hot_cdf.png' % plots_ws)
        ##if show_flag: plt.show()
        ##plt.close()
        ##
        ##
        ###### CDF by area
        ##fm_data = np.insert(np.sort(data[data['area']=='FALLON/MASON']['kc_cld_pct']), 0, 0)
        ##cm_data = np.insert(np.sort(data[data['area']=='CARSON/MASON']['kc_cld_pct']), 0, 0)
        ##ms_data = np.insert(np.sort(data[data['area']=='MASON/SMITH']['kc_cld_pct']), 0, 0)
        ##plt.figure()
        ##plt.plot(all_cold_data, prob_array(all_cold_data), 'k-', label='All')
        ##plt.plot(fm_data, prob_array(fm_data), label='Fallon/Mason')
        ##plt.plot(cm_data, prob_array(cm_data), label='Carson/Mason')
        ##plt.plot(ms_data, prob_array(ms_data), label='Mason/Smith')
        ##plt.legend(loc=4)
        ##plt.ylim(0,1)
        ##plt.title('Cold Calibration by Area')
        ##plt.ylabel('Probability')
        ##plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
        ##if save_flag: plt.savefig('%s\\cold_cdf_area.png' % plots_ws)
        ##if show_flag: plt.show()
        ##plt.close()
        ##
        ##fm_data = np.insert(np.sort(data[data['area']=='FALLON/MASON']['kc_hot_pct']), 0, 0)
        ##cm_data = np.insert(np.sort(data[data['area']=='CARSON/MASON']['kc_hot_pct']), 0, 0)
        ##ms_data = np.insert(np.sort(data[data['area']=='MASON/SMITH']['kc_hot_pct']), 0, 0)
        ##plt.figure()
        ##plt.plot(all_hot_data, prob_array(all_hot_data), 'k-', label='All')
        ##plt.plot(fm_data, prob_array(fm_data), label='Fallon/Mason')
        ##plt.plot(cm_data, prob_array(cm_data), label='Carson/Mason')
        ##plt.plot(ms_data, prob_array(ms_data), label='Mason/Smith')
        ##plt.legend(loc=4)
        ##plt.ylim(0,1)
        ##plt.title('Hot Calibration by Area')
        ##plt.ylabel('Probability')
        ##plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
        ##if save_flag: plt.savefig('%s\\hot_cdf_area.png' % plots_ws)
        ##if show_flag: plt.show()
        ##plt.close()


        ###### Separate by user
        ##plt.figure()
        ##plt.plot(all_cold_data, prob_array(all_cold_data), 'k-', label='All')
        ##for user in np.unique(data.user):
        ##    data_sub = data[data['user']==user]
        ##    user_data = np.insert(np.sort(data_sub['kc_cld_pct']), 0, 0)
        ##    plt.plot(user_data, prob_array(user_data), label=user)
        ##plt.legend(loc=4)
        ##plt.ylim(0,1)
        ##plt.title('Cold Calibration by User')
        ##plt.ylabel('Probability')
        ##plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
        ##if save_flag: plt.savefig('%s\\cold_cdf_user.png' % plots_ws)
        ##if show_flag: plt.show()
        ##plt.close()
        ##
        ##plt.figure()
        ##plt.plot(all_hot_data, prob_array(all_hot_data), 'k-', label='All')
        ##for user in np.unique(data.user):
        ##    data_sub = data[data['user']==user]
        ##    user_data = np.insert(np.sort(data_sub['kc_cld_pct']), 0, 0)
        ##    plt.plot(user_data, prob_array(user_data), label=user)
        ##plt.legend(loc=4)
        ##plt.ylim(0,1)
        ##plt.title('Hot Calibration by User')
        ##plt.ylabel('Probability')
        ##plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
        ##if save_flag: plt.savefig('%s\\hot_cdf_user.png' % plots_ws)
        ##if show_flag: plt.show()
        ##plt.close()
        ##
        ##
        ###### Separate by DOY range
        ##import matplotlib.cm as cm
        ######doy_ranges = [(1,60),(61,151),(201,300),(301,366)]
        ####doy_ranges = [(1,60),(61,151),(152,243),(244,366)]
        ##doy_ranges = [(61,90),(91,120),(121,150),(151,180),(181,210),
        ##              (211,240),(241,270),(271,300),(301,330),(331,366)]
        ##n = len(doy_ranges)
        ##
        ##plt.figure()
        ##plt.plot(all_cold_data, prob_array(all_cold_data), 'k-', label='All')
        ##for i, (doy_min, doy_max) in enumerate(doy_ranges):
        ##    data_sub = data[np.logical_and(data['doy']>=doy_min, data['doy']<=doy_max)]
        ##    user_data = np.insert(np.sort(data_sub['kc_cld_pct']), 0, 0)
        ##    plt.plot(user_data, prob_array(user_data), c=cm.hsv(1.*i/n), 
        ##             label=("%d - %d" % (doy_min, doy_max)))
        ##plt.legend(loc=4)
        ##plt.ylim(0,1)
        ##plt.title('Cold Calibration by DOY Range')
        ##plt.ylabel('Probability')
        ##plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
        ##if save_flag: plt.savefig('%s\\cold_cdf_doy.png' % plots_ws)
        ##if show_flag: plt.show()
        ##plt.close()
        ##
        ##plt.figure()
        ##plt.plot(all_hot_data, prob_array(all_hot_data), 'k-', label='All')
        ##for i, (doy_min, doy_max) in enumerate(doy_ranges):
        ##    data_sub = data[np.logical_and(data['doy']>=doy_min, data['doy']<=doy_max)]
        ##    user_data = np.insert(np.sort(data_sub['kc_cld_pct']), 0, 0)
        ##    plt.plot(user_data, prob_array(user_data), c=cm.hsv(1.*i/n), 
        ##             label=("%d - %d" % (doy_min, doy_max)))
        ##plt.legend(loc=4)
        ##plt.ylim(0,1)
        ##plt.title('Hot Calibration by DOY Range')
        ##plt.ylabel('Probability')
        ##plt.xlabel('Percentage of agricultural pixels outside calibration threshold')
        ##if save_flag: plt.savefig('%s\\hot_cdf_doy.png' % plots_ws)
        ##if show_flag: plt.show()
        ##plt.close()
        ##
    except:
        logging.exception("Unhandled Exception Error\n\n")
        raw_input("Press ENTER to continue")


################################################################################
if __name__ == '__main__':
    from sys import argv
    workspace = os.getcwd()

    #### Create Basic Logger
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    #### Run Information    
    logging.info("\n%s" % ("#"*80))
    logging.info("%-20s %s" % ("Run Time Stamp:", datetime.now().isoformat(' ')))
    logging.info("%-20s %s" % ("Current Directory:", os.getcwd()))
    logging.info("%-20s %s" % ("Script:", os.path.basename(argv[0])))

    etrf_training_plots(workspace)

    
