#--------------------------------
# Name:         et_numpy.py
# Purpose:      NumPy ET functions
#--------------------------------

# import logging
import math
import drigo

import numpy as np
import numexpr as ne

import et_common


def cos_theta_spatial_func(time, doy, dr, lat, lon):
    """

    Parameters
    ----------
    time
    doy
    dr
    lon
    lat

    Returns
    -------

    """
    sc = et_common.seasonal_correction_func(doy)
    delta = et_common.delta_func(doy)
    omega = et_common.omega_func(et_common.solar_time_rad_func(lon, time, sc))

    cos_theta = ((math.sin(delta) * np.sin(lat)) +
                 (math.cos(delta) * np.cos(lat) * np.cos(omega)))
    return cos_theta


def cos_theta_mountain_func(time, doy, dr, lon, lat, slope, aspect):
    """

    Parameters
    ----------
    time
    doy
    dr
    lon
    lat
    slope
    aspect

    Returns
    -------
    ndarray

    """

    # Convert slope (degrees to radians for calculation)
    slope = slope * (math.pi / 180.0)
    # Convert aspect (degrees to radians for calculation)
    aspect = aspect * (math.pi / 180.0)

    sc = et_common.seasonal_correction_func(doy)
    delta = et_common.delta_func(doy)
    omega = et_common.omega_func(et_common.solar_time_rad_func(lon, time, sc))
    sin_omega = np.sin(omega)
    cos_omega = np.cos(omega)
    del omega
    sin_slope = np.sin(slope)
    cos_slope = np.cos(slope)
    # Aspect is 0 as north, function is expecting 0 as south
    sin_aspect = np.sin(aspect - math.pi)
    cos_aspect = np.cos(aspect - math.pi)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    cos_theta_unadjust_array = (
        (math.sin(delta) * sin_lat * cos_slope) -
        (math.sin(delta) * cos_lat * sin_slope * cos_aspect) +
        (math.cos(delta) * cos_lat * cos_slope * cos_omega) +
        (math.cos(delta) * sin_lat * sin_slope * cos_aspect * cos_omega) +
        (math.cos(delta) * sin_aspect * sin_slope * sin_omega))
    del sin_lat, cos_lat, sin_slope
    del sin_aspect, cos_aspect, sin_omega, cos_omega

    cos_theta_array = np.maximum(
        (cos_theta_unadjust_array / cos_slope), 0.1)
    del cos_slope
    return cos_theta_array


# DEADBEEF - Trying to reduce memory usage in calculation
# def cos_theta_mountain_func(time, doy, dr, lon, lat, slope, aspect):
#     """
#
#     Parameters
#     ----------
#     time
#     doy
#     dr
#     lon
#     lat
#     slope
#     aspect
#
#     Returns
#     -------
#
#     """
#     cos_theta_array = 0
#     # Term 1 (sin(Delta)*sin(Latitude)*cos(Slope))
#     temp_array = math.sin(delta)
#     temp_array *= np.sin(lat)
#     temp_array *= np.cos(slope)
#     temp_array *= np.cos(aspect)
#     temp_array *= np.cos(omega)
#     cos_theta_array += temp_array
#     del temp_array
#     # Term 2 (-sin(Delta)*cos(Latitude)*sin(Slope)*cos(Aspect))
#     temp_array = math.sin(delta)
#     temp_array *= np.cos(lat)
#     temp_array *= np.sin(slope)
#     temp_array *= np.cos(aspect
#     cos_theta_array -= temp_array
#     del temp_array
#     # Term 3 (+cos(Delta)*cos(Latitude)*cos(Slope)*cos(Omega))
#     temp_array = math.cos(delta)
#     temp_array *= np.cos(lat)
#     temp_array *= np.cos(slope)
#     temp_array *= np.cos(omega)
#     cos_theta_array += temp_array
#     del temp_array
#     # Term 4 (+cos(Delta)*sin(Latitude)*sin(Slope)*cos(Aspect)*cos(Omega))
#     temp_array = math.cos(delta)
#     temp_array *= np.sin(lat)
#     temp_array *= np.sin(slope)
#     temp_array *= np.cos(aspect)
#     temp_array *= np.cos(omega)
#     cos_theta_array += temp_array
#     del temp_array
#     # Term 5 (+cos(Delta)*sin(Slope)*sin(Aspect)*sin(Omega))
#     temp_array = math.cos(delta)
#     temp_array *= np.sin(slope)
#     temp_array *= np.sin(aspect)
#     temp_array *= np.sin(omega)
#     cos_theta_array += temp_array
#     del temp_array
#     # Adjust
#     cos_theta_array /= np.cos(slope)
#     cos_theta_array = np.maximum(
#         cos_theta_array, 0.1, dtype=np.float32)
#     #  ((sin(Delta)*sin(Latitude)*cos(Slope))
#     #  -(sin(Delta)*cos(Latitude)*sin(Slope)*cos(Aspect))
#     #  +(cos(Delta)*cos(Latitude)*cos(Slope)*cos(Omega))
#     #  +(cos(Delta)*sin(Latitude)*sin(Slope)*cos(Aspect)*cos(Omega))
#     #  +(cos(Delta)*sin(Slope)*sin(Aspect)*sin(Omega)))
#     # cos_theta_array = (
#     #     (sin_delta * sin_lat * cos_slope) -
#     #     (sin_delta * cos_lat * sin_slope * cos_aspect) +
#     #     (cos_delta * cos_lat * cos_slope * cos_omega) +
#     #     (cos_delta * sin_lat * sin_slope * cos_aspect * cos_omega) +
#     #     (cos_delta * sin_slope * sin_aspect * sin_omega))
#     # del sin_lat, cos_lat, sin_slope
#     # del sin_aspect, cos_aspect, sin_omega, cos_omega
#     # cos_theta_array /= cos_slope
#     # del cos_slope
#     # cos_theta_array = np.maximum(
#     #     cos_theta_array, 0.1, dtype=np.float32)
#     return cos_theta_array


def l457_refl_toa_func(dn, cos_theta, dr, esun,
                       lmin, lmax, qcalmin, qcalmax,
                       band_toa_sur_mask):
    """Calculate Landsat 4, 5, or 7 TOA reflectance for all bands

    Parameters
    ----------
    dn : array_like
        Landsat raw digital number values
    cos_theta
    dr
    esun
    lmin
    lmax
    qcalmin
    qcalmax
    band_toa_sur_mask

    Returns
    -------
    ndarray

    References
    ----------
    .. [1] Chander, G., Markham, B., & Helder, D. (2009). Summary of current
       radiometric calibration coefficients for Landsat MSS, TM, ETM+, and EO-1
       ALI sensors. Remote Sensing of Environment, 113(5)
       https://doi.org/10.1016/j.rse.2009.01.007

    """
    refl_toa = np.copy(dn).astype(np.float64)
    refl_toa -= qcalmin
    refl_toa *= ((lmax - lmin) / (qcalmax - qcalmin))
    refl_toa += lmin
    refl_toa /= esun
    refl_toa[:, :, band_toa_sur_mask] /= cos_theta[
        :, :, np.newaxis].repeat(band_toa_sur_mask.size, 2)
    refl_toa[:, :, band_toa_sur_mask] *= (math.pi / dr)
    # Don't clip thermal band since it is not scaled from 0-1
    refl_toa[:, :, band_toa_sur_mask] = np.clip(
        refl_toa[:, :, band_toa_sur_mask], 0.0001, 1)
    return refl_toa.astype(np.float32)


def l457_refl_toa_band_func(dn, cos_theta, dr, esun,
                            lmin, lmax, qcalmin, qcalmax):
    """Landsat 4, 5, or 7 DN -> TOA reflectance (single band)

    Parameters
    ----------
    dn : array_like
        Landsat raw digital number values
    cos_theta
    dr
    esun
    lmin
    lmax
    qcalmin
    qcalmax

    Returns
    -------
    ndarray

    References
    ----------
    .. [1] Chander, G., Markham, B., & Helder, D. (2009). Summary of current
       radiometric calibration coefficients for Landsat MSS, TM, ETM+, and EO-1
       ALI sensors. Remote Sensing of Environment, 113(5)
       https://doi.org/10.1016/j.rse.2009.01.007

    """
    refl_toa = np.copy(dn).astype(np.float64)
    refl_toa -= qcalmin
    refl_toa *= ((lmax - lmin) / (qcalmax - qcalmin))
    refl_toa += lmin
    refl_toa /= cos_theta
    refl_toa *= (math.pi / (dr * esun))
    np.clip(refl_toa, 0.0001, 1, out=refl_toa)
    return refl_toa.astype(np.float32)


def l457_ts_bt_band_func(dn, lmin, lmax, qcalmin, qcalmax, k1, k2):
    """Landsat 4, 5, or 7 DN -> brightness temperature (single band)

    Parameters
    ----------
    dn : ndarray
    lmin : array_like
    lmax : array_like
    qcalmin : array_like
    qcalmax : array_like
    k1 : float
    k2 : float

    Returns
    -------
    ndarray

    References
    ----------
    .. [1] Chander, G., Markham, B., & Helder, D. (2009). Summary of current
       radiometric calibration coefficients for Landsat MSS, TM, ETM+, and EO-1
       ALI sensors. Remote Sensing of Environment, 113(5)
       https://doi.org/10.1016/j.rse.2009.01.007

    """
    ts_bt = np.copy(dn).astype(np.float64)
    ts_bt -= qcalmin
    ts_bt *= ((lmax - lmin) / (qcalmax - qcalmin))
    ts_bt += lmin
    return ts_bt_func(ts_bt, k1, k2).astype(np.float32)


def l8_refl_toa_band_func(dn, cos_theta, refl_mult, refl_add):
    """Landsat 8 DN -> TOA reflectance (single band)

    Parameters
    ----------
    dn : ndarray
    cos_theta : array_like
    refl_mult : array_like
        Reflectance multiplicative scaling factors
    refl_add : array_like
        Reflectance additive scaling factors

    Returns
    -------
    ndarray

    References
    ----------
    .. [1] Landsat 8 Data Users Handbook
       https://landsat.usgs.gov/landsat-8-l8-data-users-handbook

    """
    refl_toa = np.copy(dn).astype(np.float64)
    refl_toa *= refl_mult
    refl_toa += refl_add
    refl_toa /= cos_theta
    np.clip(refl_toa, 0.0001, 1, out=refl_toa)
    return refl_toa


def l8_ts_bt_band_func(dn, rad_mult, rad_add, k1, k2):
    """Landsat 8 -> brightness temperature (single band)

    Parameters
    ----------
    dn
    rad_mult
    rad_add
    k1
    k2

    Returns
    -------
    ndarray

    References
    ----------
    .. [1] Landsat 8 Data Users Handbook
       https://landsat.usgs.gov/landsat-8-l8-data-users-handbook

    """
    ts_bt = np.copy(dn).astype(np.float64)
    ts_bt *= rad_mult
    ts_bt += rad_add
    return ts_bt_func(ts_bt, k1, k2).astype(np.float32)


def bqa_fmask_func(qa):
    """Construct Fmask array from Landsat Collection 1 TOA QA array

    Parameters
    ----------
    qa : ndarray

    Returns
    -------
    ndarray

    Notes
    -----
    https://landsat.usgs.gov/collectionqualityband
    https://code.earthengine.google.com/356a3580096cca315785d0859459abbd

    Confidence values:
    00 = "Not Determined" = Algorithm did not determine the status of this condition
    01 = "No" = Algorithm has low to no confidence that this condition exists
        (0-33 percent confidence)
    10 = "Maybe" = Algorithm has medium confidence that this condition exists
        (34-66 percent confidence)
    11 = "Yes" = Algorithm has high confidence that this condition exists
        (67-100 percent confidence

    """
    # Extracting cloud masks from BQA using np.right_shift() and np.bitwise_and()
    # Cloud (med & high confidence), then snow, then shadow, then fill
    # Low confidence clouds tend to be the FMask buffer
    fill_mask = np.bitwise_and(np.right_shift(qa, 0), 1) >= 1
    cloud_mask = np.bitwise_and(np.right_shift(qa, 4), 1) >= 1  # cloud bit
    cloud_mask &= np.bitwise_and(np.right_shift(qa, 5), 3) >= 2  # cloud conf.
    cloud_mask |= np.bitwise_and(np.right_shift(qa, 11), 3) >= 3  # cirrus
    shadow_mask = np.bitwise_and(np.right_shift(qa, 7), 3) >= 3
    snow_mask = np.bitwise_and(np.right_shift(qa, 9), 3) >= 3

    fmask = (fill_mask != True).astype(np.uint8)
    fmask[shadow_mask] = 2
    fmask[snow_mask] = 3
    fmask[cloud_mask] = 4

    return fmask


def tau_broadband_func(pair, w, cos_theta, kt=1):
    """Broadband transmittance

    Parameters
    ----------
    pair : array_like
        Air pressure [kPa].
    w : array_like
        Precipitable water in the atmosphere [mm]
    cos_theta : array_like
    kt : float

    Returns
    -------
    ndarray

    References
    ----------

    """

    tau_broadband = tau_direct_func(pair, w, cos_theta, kt)
    tau_broadband += tau_diffuse_func(tau_broadband)
    return tau_broadband.astype(np.float32)


def tau_direct_func(pair, w, cos_theta, kt=1):
    """

    Parameters
    ----------
    pair : array_like
    w : array_like
    cos_theta : array_like
    kt : float

    Returns
    -------
    ndarray

    Notes
    -----
    0.98 * np.exp((-0.00146 * pair / kt) - (0.075 * np.power(w, 0.4)))

    References
    ----------
    .. [1] Tasumi, M., Allen, R., and Trezza, R. (2008). At-surface reflectance
       and albedo from satellite for operational calculation of land surface
       energy balance. Journal of Hydrologic Engineering 13(2):51-63.
       https://doi.org/10.1061/(ASCE)1084-0699(2008)13:2(51)

    """
    t1 = np.copy(pair).astype(np.float64)
    t1 /= kt
    t1 *= -0.00146
    t1 /= cos_theta
    t2 = np.copy(w).astype(np.float64)
    t2 /= cos_theta
    np.power(t2, 0.4, out=t2)
    t2 *= 0.075
    t1 -= t2
    del t2
    np.exp(t1, out=t1)
    t1 *= 0.98
    return t1


def tau_diffuse_func(tau_direct):
    """

    Parameters
    ----------
    tau_direct : array_like

    Returns
    -------
    ndarray

    Notes
    -----
    Model differs from formulas in METRIC manual.
    Eqn is not applied, per Rick Allen it is not needed.

    References
    ----------
    .. [1] Tasumi, M., Allen, R., and Trezza, R. (2008). At-surface reflectance
       and albedo from satellite for operational calculation of land surface
       energy balance. Journal of Hydrologic Engineering 13(2):51-63.
       https://doi.org/10.1061/(ASCE)1084-0699(2008)13:2(51)

    """
    tau = np.copy(tau_direct).astype(np.float64)
    tau *= -0.36
    tau += 0.35
    return tau
    # return np.where(tau_direct_array >= 0.15),
    #                 (0.35-0.36*tau_direct_array),
    #                 (0.18-0.82*tau_direct_array))


def tau_narrowband_func(pair, w, cos_theta, kt, c1, c2, c3, c4, c5):
    """Narrowband transmittance

    Parameters
    ----------
    pair : array_like
        Air pressure [kPa].
    w : array_like
        Precipitable water in the atmosphere [mm]
    cos_theta : array_like
    kt : float
    c1 : float
    c2 : float
    c3 : float
    c4 : float
    c5 : float

    Returns
    -------
    ndarray

    Notes
    -----
    IN:  c1 * exp(((c2*pair) / (kt*cos_theta)) - ((c3*w+c4) / cos_theta)) + c5
    OUT: c1 * exp(((c2*pair) / (kt*1.0)) - ((c3*w+c4) / 1.0)) + c5

    References
    ----------
    .. [1] Tasumi, M., Allen, R., and Trezza, R. (2008). At-surface reflectance
       and albedo from satellite for operational calculation of land surface
       energy balance. Journal of Hydrologic Engineering 13(2):51-63.
       https://doi.org/10.1061/(ASCE)1084-0699(2008)13:2(51)

    """
    t1 = np.copy(pair).astype(np.float64)
    t1 /= kt
    t1 *= c2
    t2 = np.copy(w)
    t2 *= c3
    t2 += c4
    t1 -= t2
    del t2
    t1 /= cos_theta
    np.exp(t1, out=t1)
    t1 *= c1
    t1 += c5
    return t1.astype(np.float32)


def refl_sur_tasumi_func(refl_toa, pair, w, cos_theta, kt,
                         c1, c2, c3, c4, c5, cb, band_cnt):
    """Tasumi at-surface reflectance

    Parameters
    ----------
    refl_toa : ndarray
        Top-of-atmosphere reflectance.
    pair : array_like
        Air pressure [kPa].
    w : array_like
        Precipitable water in the atmosphere [mm]
    cos_theta : array_like
    kt : float
        Clearness coefficient.
    c1 : float
    c2 : float
    c3 : float
    c4 : float
    c5 : float
    cb : float
    band_cnt : int

    Returns
    -------
    ndarray

    Notes
    -----
    refl_sur = (refl_toa - cb * (1 - tau_in)) / (tau_in * tau_out)

    References
    ----------
    .. [1] Tasumi, M., Allen, R., and Trezza, R. (2008). At-surface reflectance
       and albedo from satellite for operational calculation of land surface
       energy balance. Journal of Hydrologic Engineering 13(2):51-63.
       https://doi.org/10.1061/(ASCE)1084-0699(2008)13:2(51)

    """
    if np.all(np.isnan(refl_toa)):
        return refl_toa

    # Reshape arrays to match the surface reflectance arrays
    pair_mod = pair[:, :, np.newaxis].repeat(band_cnt, 2).astype(np.float64)
    w_mod = w[:, :, np.newaxis].repeat(band_cnt, 2).astype(np.float64)
    cos_theta_mod = cos_theta[:, :, np.newaxis].repeat(band_cnt, 2).astype(np.float64)

    tau_in = tau_narrowband_func(
        pair_mod, w_mod, cos_theta_mod, kt, c1, c2, c3, c4, c5)
    tau_out = tau_narrowband_func(
        pair_mod, w_mod, 1, kt, c1, c2, c3, c4, c5)
    del cos_theta_mod, pair_mod, w_mod
    refl_sur = np.copy(tau_in)
    refl_sur *= -1
    refl_sur += 1
    refl_sur *= -cb
    refl_sur += refl_toa
    refl_sur /= tau_in
    refl_sur /= tau_out
    np.clip(refl_sur, 0.0001, 1, out=refl_sur)

    return refl_sur.astype(np.float32)


def albedo_sur_func(refl_sur, wb):
    """Tasumi at-surface albedo

    Parameters
    ----------
    refl_sur : ndarray
    wb :

    Returns
    -------
    ndarray

    References
    ----------
    .. [1] Tasumi, M., Allen, R., and Trezza, R. (2008). At-surface reflectance
       and albedo from satellite for operational calculation of land surface
       energy balance. Journal of Hydrologic Engineering 13(2):51-63.
       https://doi.org/10.1061/(ASCE)1084-0699(2008)13:2(51)

    """
    return np.sum(refl_sur * wb, axis=2)


def albedo_ts_corrected_func(albedo_sur, ndvi_toa, ts_array, hot_px_temp,
                             cold_px_temp, k_value, dense_veg_min_albedo):
    """Updated Ts based on METRIC Manual - Eqn. 16-1"""
    masked = (albedo_sur < dense_veg_min_albedo) & (ndvi_toa > 0.45)
    ts_array[masked] = ts_array[masked] + ((dense_veg_min_albedo  - albedo_sur[masked]) * k_value * ((hot_px_temp-cold_px_temp) * (0.95)))

    """Updated albedo based on METRIC Manual - Eqn. 16-1"""
    masked = (albedo_sur < dense_veg_min_albedo) & (ndvi_toa > 0.6)
    albedo_sur[masked] = dense_veg_min_albedo
    masked = (albedo_sur < dense_veg_min_albedo) & ((ndvi_toa > 0.4) & (ndvi_toa < 0.6))
    albedo_sur[masked] = dense_veg_min_albedo - (dense_veg_min_albedo - albedo_sur[masked]) * (1 - ((ndvi_toa[masked] - 0.4) / (0.6 - 0.4)))
    return ts_array, albedo_sur


# Vegetation Indices
def ndi_func(a, b, l=0.0):
    """Normalized difference index function

    Parameters
    ----------
    a : array_like
    b : array_like
    l :

    Returns
    -------
    array_like

    Notes
    -----
    Function can be used to calculate SAVI ([1]_, [2]_) by setting l != 0.

    References
    ----------
    .. [1] Huete, A. (1988). A soil-adjusted vegetation index (SAVI).
       Remote Sensing of Environment, 25(3).
       https://doi.org/10.1016/0034-4257(88)90106-X
    .. [2] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    ndi = ((1. + l) * (a - b) / (l + a + b))
    # Manually set output value when a and b are zero
    # ndi[((l+a+b) != 0)] = 0
    return ndi


def savi_lai_func(savi):
    """Compute leaf area index (LAI) from SAVI

    Parameters
    ----------
    savi : array_like
        Soil adjusted vegetation index.

    Returns
    -------
    ndarray

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    return np.clip((11. * np.power(savi, 3)), 0, 6)


def ndvi_lai_func(ndvi):
    """Compute leaf area index (LAI) from NDVI

    Parameters
    ----------
    ndvi : array_like
        Normalized difference vegetation index.

    Returns
    -------
    ndarray

    References
    ----------
    .. [1] Trezza and Allen 2014?

    """
    return np.clip((7. * np.power(ndvi, 3)), 0, 6)


def ratio_func(a, b):
    """Compute ratio of two values

    Parameters
    ----------
    a : array_like
    b : array_like

    Returns
    -------
    array_like

    """
    return a / b


def evi_func(blue, red, nir):
    """Compute enhanced vegetation index

    Parameters
    ----------
    blue : array_like
        Blue band (band 1 on Landsat 5/7, band 2 on Landsat 8).
    red :
        Red band (band 3 on Landsat 5/7, band 4 on Landsat 8).
    nir : array_like
        Near infrared band (band 4 on Landsat 5/7, band 5 on Landsat 8).

    Returns
    -------
    array_like

    References
    ----------
    .. [1] Huete et al. (2002).
       Overview of the radiometric and biophysical performance of the MODIS
       vegetation indices. Remote Sensing of Environment, 83.
       https://doi.org/10.1016/S0034-4257(02)00096-2

    """
    return (2.5 * (nir - red)) / (nir + 6 * red - 7.5 * blue + 1)


def tc_bright_func(reflectance, image_type='TOA'):
    """Tasseled cap brightness

    Parameters
    ----------
    reflectance : array_like
        Reflectance.
    image_type : {'TOA' (default), 'SUR'}, optional
        Reflectance type.

    Returns
    -------
    ndarray

    References
    ----------
    DEADBEEF - Check these URLs and generate actual references and copy
    to all functions
    LT04/LT05 - http://www.gis.usu.edu/~doug/RS5750/assign/OLD/RSE(17)-301.pdf
    LE07 - http://landcover.usgs.gov/pdf/tasseled.pdf
    LC08 - http://www.tandfonline.com/doi/abs/10.1080/2150704X.2014.915434
    https://www.researchgate.net/publication/262005316_Derivation_of_a_tasselled_cap_transformation_based_on_Landsat_8_at-_satellite_reflectance

    """
    if image_type == 'SUR':
        tc_bright = np.array([0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0, 0.2303])
    elif image_type == 'TOA':
        tc_bright = np.array([0.3561, 0.3972, 0.3904, 0.6966, 0.2286, 0, 0.1596])
    return np.sum(reflectance * tc_bright, axis=2).astype(np.float32)


def tc_green_func(reflectance, image_type='TOA'):
    """Tasseled cap greenness

    Parameters
    ----------
    reflectance : array_like
    image_type : {'TOA' (default), 'SUR'}, optional
        Reflectance type.

    Returns
    -------
    ndarray

    References
    ----------


    """
    if image_type == 'SUR':
        tc_green = np.array([-0.1063, -0.2819, -0.4934, 0.7940, -0.0002, 0, -0.1446])
    elif image_type == 'TOA':
        tc_green = np.array([-0.3344, -0.3544, -0.4556, 0.6966, -0.0242, 0, -0.2630])
    return np.sum(reflectance * tc_green, axis=2).astype(np.float32)


def tc_wet_func(reflectance, image_type='TOA'):
    """Tasseled cap wetness

    Parameters
    ----------
    reflectance : array_like
    image_type : {'TOA' (default), 'SUR'}, optional
        Reflectance type.

    Returns
    -------
    ndarray

    References
    ----------


    """
    if image_type == 'SUR':
        tc_wet = np.array([
            0.0315, 0.2021, 0.3102, 0.1594, -0.6806, 0, -0.6109])
    elif image_type == 'TOA':
        tc_wet = np.array([
            0.2626, 0.2141, 0.0926, 0.06564, -0.7629, 0, -0.5388])
    return np.sum(reflectance * tc_wet, axis=2).astype(np.float32)


def etstar_func(evi, etstar_type='mean'):
    """Compute ET*

    Parameters
    ----------
    evi : array_like
        Enhanced vegetation index.
    etstar_type : {'mean', 'lpi', 'upi', 'lci', 'uci'}, optional

    Returns
    -------
    ndarray

    References
    ----------
    .. [1] Beamer, J., Huntington, J., Morton, C., & Pohll, G. (2011).
       Estimating annual groundwater evapotranspiration from phreatophytes in
       the Great Basin using Landsat and flux tower measurements.
       Journal of the American Water Resources Association, 49(3).
       https://doi.org/10.1111/jawr.12058

    """
    c_dict = dict()
    c_dict['mean'] = np.array([-0.1955, 2.9042, -1.5916]).astype(np.float32)
    c_dict['lpi'] = np.array([-0.2871, 2.9192, -1.6263]).astype(np.float32)
    c_dict['upi'] = np.array([-0.1039, 2.8893, -1.5569]).astype(np.float32)
    c_dict['lci'] = np.array([-0.2142, 2.9175, -1.6554]).astype(np.float32)
    c_dict['uci'] = np.array([-0.1768, 2.891, -1.5278]).astype(np.float32)
    try:
        c = c_dict[etstar_type]
    except KeyError:
        raise SystemExit()

    # ET* calculation
    etstar = np.copy(evi)
    etstar *= c[2]
    etstar += c[1]
    etstar *= evi
    etstar += c[0]
    np.maximum(etstar, 0., out=etstar)
    return etstar


def etstar_etg_func(etstar, eto, ppt):
    """Computed ET* based groundwater evapotranspiration (ETg)

    Parameters
    ----------
    etstar : array_like
        ET*
    eto : array_like
        Reference ET [mm].
    ppt : array_like
        Precipitation [mm].

    Returns
    -------
    ndarray

    """
    return np.copy(etstar) * (eto - ppt)


def etstar_et_func(etstar, eto, ppt):
    """Compute ET* based evapotranspiration (ET)

    Parameters
    ----------
    etstar : array_like
        ET*
    eto : array_like
        Reference ET [mm]
    ppt : array_like
        Precipitation [mm]

    Returns
    -------
    ndarray

    """
    return np.copy(etstar) * (eto - ppt) + ppt


def em_nb_func(lai, water_index, water_threshold=0):
    """Narrowband emissivity

    Parameters
    ----------
    lai : array_like
        Leaf area index
    water_index : array_like
        Normalized index used to identify water pixels (typically NDVI).
    water_threshold : float, optional
        Pixels with water_index values less than this value will have the water
        emissivity value applied.

    Returns
    -------
    ndarray

    Notes
    -----
    em_0 = (0.97 + (lai / 300.)) for LAI <= 3
    em_0 = 0.98 for LAI > 3
    em_0 = 0.985 for water
    DEADBEEF - Check 0.99 value in code for water

    References
    ----------
    .. [1] Tasumi, M. (2003). Progress in operational estimation of regional
       evapotranspiration using satellite imagery. Ph.D. dissertation.
    .. [2] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    em_nb = np.copy(lai).astype(np.float32)
    em_nb /= 300.
    em_nb += 0.97
    em_nb[(water_index > water_threshold) & (lai > 3)] = 0.98
    em_nb[water_index < water_threshold] = 0.99
    return em_nb


def em_0_func(lai, water_index, water_threshold=0):
    """Broadband emissivity

    Parameters
    ----------
    lai : array_like
        Leaf area index.
    water_index : array_like
        Normalized index used to identify water pixels (typically NDVI).
    water_threshold : float, optional
        Pixels with water_index values less than this value will have the water
        emissivity value applied.

    Returns
    -------
    ndarray

    Notes
    -----
    em_0 = (0.95 + (lai / 100.)) for LAI <= 3
    em_0 = 0.98 for LAI > 3
    em_0 = 0.985 for water

    References
    ----------
    .. [1] Tasumi, M. (2003). Progress in operational estimation of regional
       evapotranspiration using satellite imagery. Ph.D. dissertation.
    .. [2] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    em_0 = np.copy(lai).astype(np.float32)
    em_0 /= 100.
    em_0 += 0.95
    em_0[(water_index > water_threshold) & (lai > 3)] = 0.98
    em_0[water_index <= water_threshold] = 0.985
    return em_0


def rc_func(thermal_rad, em_nb, rp, tnb, rsky):
    """Corrected Radiance

    Parameters
    ----------
    thermal_rad : array_like
        Thermal band spectral radiance [W m-2 sr-1 um-1].
    em_nb : array_like
        Narrow band emissivity.
    rp : float
        Path radiance (in the thermal band) [W m-2 sr-1 um-1].
    tnb : float
        Transmissivity of air (in the thermal band).
    rsky : float
        Clear sky downward thermal radiance [W m-2 sr-1 um-1].

    Returns
    -------
    ndarray

    Notes
    -----
    rc = ((thermal_rad - rp) / tnb) - ((1.0 - em_nb) * rsky)

    References
    ----------
    .. [1] Wukelic et al. (1989).
    .. [2] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    # DEADBEEF - Why is ndmin=1 being set here?
    rc = np.array(thermal_rad, copy=True, ndmin=1).astype(np.float64)
    # rc = np.copy(thermal_rad_toa).astype(np.float32)
    rc -= rp
    rc /= tnb
    rc -= rsky
    rc += (em_nb * rsky)
    return rc.astype(np.float32)


def ts_func(em_nb, rc, k1, k2):
    """Surface Temperature

    Parameters
    ----------
    em_nb : array_like
        Narrow band emissivity.
    rc : array_like
        Corrected thermal radiance [W m-2 sr-1 um-1].
    k1 : float
        Calibration constant.
    k2 : float
        Calibration constant.

    Returns
    -------
    ndarray

    Notes
    -----
    ts = k2 / log(((em_nb * k1) / rc) + 1.0)

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)
    .. [2] Markham and Barker (1986).

    """
    ts = np.copy(em_nb).astype(np.float64)
    ts *= k1
    ts /= rc
    ts += 1.0
    np.log(ts, out=ts)
    np.reciprocal(ts, out=ts)
    ts *= k2
    return ts.astype(np.float32)


def ts_bt_func(thermal_rad, k1, k2):
    """Calculate brightness temperature from thermal radiance

    Parameters
    ----------
    thermal_rad : array_like
        Thermal band spectral radiance [W m-2 sr-1 um-1].
    k1 : float
        Calibration constant.
    k2 : float
        Calibration constant.

    Returns
    -------
    ndarray

    Notes
    -----
    ts_bt = k2 / log((k1 / L) + 1.0)

    References
    ----------
    .. [1] Chander, G., Markham, B., & Helder, D. (2009). Summary of current
       radiometric calibration coefficients for Landsat MSS, TM, ETM+, and EO-1
       ALI sensors. Remote Sensing of Environment, 113(5)
       https://doi.org/10.1016/j.rse.2009.01.007

    """
    ts_bt = np.copy(thermal_rad).astype(np.float64)
    ts_bt[ts_bt <= 0] = np.nan
    np.reciprocal(ts_bt, out=ts_bt)
    ts_bt *= k1
    ts_bt += 1.0
    np.log(ts_bt, out=ts_bt)
    np.reciprocal(ts_bt, out=ts_bt)
    ts_bt *= k2
    return ts_bt.astype(np.float32)


def thermal_rad_func(ts_bt, k1, k2):
    """Back calculate thermal radiance from brightness temperature

    Parameters
    ----------
    ts_bt : array_like
        Brightness temperature [K].
    k1 : float
        Calibration constant.
    k2 : float
        Calibration constant.

    Returns
    -------
    ndarray

    Notes
    -----
    thermal_rad = k1 / (exp(k2 / ts_bt) - 1.0)

    References
    ----------
    .. [1] Chander, G., Markham, B., & Helder, D. (2009). Summary of current
       radiometric calibration coefficients for Landsat MSS, TM, ETM+, and EO-1
       ALI sensors. Remote Sensing of Environment, 113(5)
       https://doi.org/10.1016/j.rse.2009.01.007

    """
    thermal_rad = np.copy(ts_bt).astype(np.float64)
    np.reciprocal(thermal_rad, out=thermal_rad)
    thermal_rad *= k2
    np.exp(thermal_rad, out=thermal_rad)
    thermal_rad -= 1.0
    np.reciprocal(thermal_rad, out=thermal_rad)
    thermal_rad *= k1
    return thermal_rad.astype(np.float32)


def lapse_func(elevation, datum, lapse_elev, lapse_flat, lapse_mtn):
    """Calculates and returns lapse component only for lapsing/ delapsing surface temperature based on elevation"""
    ts_a = np.copy(elevation).astype(np.float64)
    ts_a -= datum
    ts_a *= (lapse_flat * 0.001)
    ts_b = np.copy(elevation).astype(np.float64)
    ts_b -= lapse_elev
    ts_b *= (lapse_mtn * 0.001)
    ts_b += ((lapse_elev - datum) * lapse_flat * 0.001)
    return np.where(elevation <= lapse_elev, ts_a, ts_b).astype(np.float32)


def ts_lapsed_func(ts, elevation, datum, lapse_elev, lapse_flat, lapse_mtn):
    """Lapse surface temperature based on elevation"""
    ts_a = np.copy(elevation).astype(np.float64)
    ts_a -= datum
    ts_a *= (lapse_flat * - 0.001)
    ts_a += ts
    ts_b = np.copy(elevation).astype(np.float64)
    ts_b -= lapse_elev
    ts_b *= (lapse_mtn * - 0.001)
    ts_b += ((lapse_elev - datum) * lapse_flat * -0.001)
    ts_b += ts
    return np.where(elevation <= lapse_elev, ts_a, ts_b).astype(np.float32)


def ts_delapsed_func(ts, elevation, datum, lapse_elev, lapse_flat, lapse_mtn):
    """Delapse surface temperature based on elevation"""
    ts_a = np.copy(elevation).astype(np.float64)
    ts_a -= datum
    ts_a *= (lapse_flat * 0.001)
    ts_a += ts
    ts_b = np.copy(elevation).astype(np.float64)
    ts_b -= lapse_elev
    ts_b *= (lapse_mtn * 0.001)
    ts_b += ((lapse_elev - datum) * lapse_flat * 0.001)
    ts_b += ts
    return np.where(elevation <= lapse_elev, ts_a, ts_b).astype(np.float32)


def ts_dem_dry_func(ts_dem_cold, ts_dem_hot, kc_cold, kc_hot):
    """"""
    return (ts_dem_hot + (kc_hot * ((ts_dem_hot - ts_dem_cold) /
                                    (kc_cold - kc_hot))))


def calculate_lst_terrain_general(lst_tall_veg, slope, aspect, sun_azimuth, temp_diff=10.0):
    """Compute the LST temperature of opposing shaded slopes.

    .. topic:: References

        - "METRIC Applications Manual - Version 3.0"
          Allen R.G., Trezza R., Tasumi M., Kjaersgaard J., (2014)

    """
    # Calculating the factor k, 0, 2PI = facing sun, PI=away from sun.

    mem_1 = (aspect - sun_azimuth) * math.pi / 180
    mem_1 = np.where(aspect < sun_azimuth, (360 - sun_azimuth + aspect) * math.pi / 180, mem_1)

    mem_2 = np.where(mem_1 < 0, mem_1 * -1, mem_1)

    lst_terrain = lst_tall_veg - temp_diff * np.cos(mem_2)
    lst_terrain = np.where(slope <= 5, lst_tall_veg, lst_terrain)

    return lst_terrain


def sin_beta_daily(lat, doy):
    """ Daily sin of the angle of the sun above the horizon (D.5 and Eqn 62 from ASCE Ref ET Allen, 2005)

    .. topic:: References

    ASCE-EWRI, 2005.
    The ASCE Standardized Reference Evapotranspiration Equation.
    ASCE, Reston, Virginia.

    Allen, R. G., Trezza, R., & Tasumi, M. (2006).
    Analytical integrated functions for daily solar radiation on slopes.
    Agricultural and Forest Meteorology, 139(1-2), 55-73.
    """

    sin_beta_24 = np.sin(
        0.85 + 0.3 * lat * np.sin(et_common.doy_fraction_func(doy) - 1.39435) -
        0.42 * np.power(lat, 2))
    sin_beta_24 = np.maximum(sin_beta_24, 0.1)
    return sin_beta_24


def rso_24_func_flat(lat, doy, pair, ea, ra):

    # This is taken from the daily reference ET calculation for daily Rso
    # This is the full clear sky solar formulation

    sin_beta_24 = sin_beta_daily(lat, doy)

    # Precipitable water (Eqn D.3)
    w = et_common.precipitable_water_func(pair, ea)

    # Clearness index for direct beam radiation (Eqn D.2)
    # Limit sin_beta >= 0.01 so that KB does not go undefined
    kb_24 = (0.98 * np.exp((-0.00146 * pair) / sin_beta_24 -
                        0.075 * np.power((w / sin_beta_24), 0.4)))

    # Transmissivity index for diffuse radiation (Eqn D.4)
    kd_24 = np.minimum(-0.36 * kb_24 + 0.35, 0.82 * kb_24 + 0.18)

    # Clear sky solar radiation (Eqn D.1)
    rso = ra * (kb_24 + kd_24)
    return rso


def rso_24_func_mountain(rso_24_flat, ra_24, lat, slope, doy, pair, ea):
    """Compute the 24 hr clear sky solar radiation for a sloping surface

    .. topic:: References

    Allen, R. G., Trezza, R., & Tasumi, M. (2006).
    Analytical integrated functions for daily solar radiation on slopes.
    Agricultural and Forest Meteorology, 139(1-2), 55-73.


    """

    # Convert slope (degrees to radians for calculation)
    slope = slope * (math.pi / 180.0)

    sin_beta_24 = sin_beta_daily(lat, doy)

    delta = et_common.delta_func(doy)
    w = et_common.precipitable_water_func(pair, ea)
    del ea
    KB_24 = tau_direct_func(pair, w, sin_beta_24)
    del w, pair, sin_beta_24
    KD_24 = tau_diffuse_func(KB_24)

    terrain_albedo = np.zeros_like(KB_24) + 0.2
    pi = math.pi

    rso_24 = ne.evaluate("KB_24*ra_24\
        +KD_24*1367.0/pi*(1+0.033*cos(2*pi/365*doy))\
        *(arccos(-tan(lat)*tan(delta))*sin(delta)*sin(lat)\
        +cos(delta)*cos(lat)*sin(arccos(-tan(lat)*tan(delta))))\
        *(0.75+0.25*cos(slope)-0.5*(slope)/pi)\
        +rso_24_flat*terrain_albedo*(1-(0.75+0.25*cos(slope)-0.5*(slope)/pi))",
                         {'rso_24_flat': rso_24_flat,
                          'ra_24': ra_24,
                          'slope': slope,
                          'delta': delta,
                          'KB_24': KB_24,
                          'KD_24': KD_24,
                          'terrain_albedo': terrain_albedo,
                          'pi': pi,
                          'doy': doy,
                          'lat': lat
                          }
                         )

    del rso_24_flat, ra_24, slope, delta, KB_24, KD_24, terrain_albedo,

    return rso_24


def daylight_hours_func(latitude, doy):
    """Compute the number of daylight hours.

    :param latitude: Latitude [radians]
    :param doy: day of year (julian date)
    :returns: daylight hours
    :rtype: ee.Image
    """
    pi = math.pi

    return ne.evaluate(
        '24.0/pi*arccos(-tan(latitude)*tan(0.409*sin(2*pi*doy/365-1.39)))')


def g_water_func(rn, acq_doy):
    """Adjust water heat storage based on day of year"""
    return rn * (-1.25 + (0.0217 * acq_doy) - (5.87E-5 * (acq_doy ** 2)))


def excess_res_func(u3):
    """Excess resistance function that can be applied for shrub and grassland

         .. topic:: References

         - "METRIC Applications Manual - Version 3.0"
           Allen R.G., Trezza R., Tasumi M., Kjaersgaard J., (2014)

    Excess res. needs to be recalculated if additional wind is applied
    """
    if u3 < 2.6:
        return 0.0
    elif u3 > 15.0:
        return 2.0
    else:
        return (
            (0.01303 * u3 ** 3) - (0.43508 * u3 ** 2) +
            (4.27477 * u3) - 8.283524)


def ra_daily_func(lat, doy):
    """Daily extraterrestrial radiation [W m -2]

    ..topic: References

    Allen, R. G., Trezza, R., & Tasumi, M. (2006).
    Analytical integrated functions for daily solar radiation on slopes.
    Agricultural and Forest Meteorology, 139(1-2), 55-73.


    Parameters
    ----------
    lat : array_like
        Latitude [radians].
    doy : array_like
        Day of year.

    Returns
    -------
    ndarray

    Notes
    -----
    This function  is only being called by et_numpy.rn_24_func().
    That function could be changed to use the refet.calcs._ra_daily() function
    instead, in which case this function could be removed.

    """

    delta = et_common.delta_func(doy)
    omegas = et_common.omega_sunset_func(lat, delta)
    theta = (omegas * np.sin(lat) * np.sin(delta) +
             np.cos(lat) * np.cos(delta) * np.sin(omegas))

    return (1367 / math.pi) * et_common.dr_func(doy) * theta


def ra_daily_mountain_func(lat, doy, slope, aspect, ra_min=0.1):
    """Daily extraterrestrial radiation for mountain terrain [W m-2]

    .. topic:: References

    Allen, R. G., Trezza, R., & Tasumi, M. (2006).
    Analytical integrated functions for daily solar radiation on slopes.
    Agricultural and Forest Meteorology, 139(1-2), 55-73.


    Parameters
    ----------
    lat : array_like
        Latitude [radians].
    doy : array_like
        Day of year.
    slope : array_like
        slope [degrees]
    aspect : array_like
        aspect [degrees]

    Returns
    -------
    ndarray

    """

    # Convert slope from degrees to radians
    slope *= math.pi / 180

    def hour_angle(time):
        return 15 * (time - 12.0) * (math.pi / 180)

    time = [(0.5 * i) + 0.25 for i in range(0, 48)]

    sun_hour_angles = [hour_angle(t) for t in time]

    # Reshape to (48, 1, 1) to allow broadcasting to work correctly with block rasters
    hour_angles = np.array(sun_hour_angles)[:, None, None]

    delta = et_common.delta_func(doy)
    dr = et_common.dr_func(doy)

    x = ne.evaluate('sin(delta)*sin(lat)*cos(slope)-sin(delta)*cos(lat)*sin(slope)*cos((aspect-180)*pi/180)', {
        'delta': delta,
        'slope': slope,
        'aspect': aspect,
        'lat': lat,
        'pi': math.pi
    })

    y = ne.evaluate('cos(delta)*cos(lat)*cos(slope)+cos(delta)*sin(lat)*sin(slope)*cos((aspect-180)*pi/180)', {
        'delta': delta,
        'slope': slope,
        'aspect': aspect,
        'lat': lat,
        'pi': math.pi
    })

    z = ne.evaluate('cos(delta)*sin(slope)*sin((aspect-180)*pi/180)', {
        'delta': delta,
        'aspect': aspect,
        'slope': slope,
        'pi': math.pi
    })

    Ra_24_1 = ne.evaluate('(x+y*cos(hour_angles)+z*sin(hour_angles))', {
        'y': y,
        'x': x,
        'hour_angles': hour_angles,
        'z': z
    })

    lim_1 = ne.evaluate('(sin(delta)*sin(lat)+cos(delta)*cos(lat)*cos(hour_angles))',
                        {
                            'lat': lat,
                            'delta': delta,
                            'hour_angles': hour_angles
                        })

    lim_2 = ne.evaluate('(x+y*cos(hour_angles)+z*sin(hour_angles))',
                        {
                            'x': x,
                            'y': y,
                            'hour_angles': hour_angles,
                            'z': z
    })

    Ra_24_1 = np.where(lim_1 <= 0, 0, Ra_24_1)
    Ra_24_1 = np.where(lim_2 <= 0, 0, Ra_24_1)

    # New array is shape (48, blocksize, blocksize) so want to sum over first axis (0) to sum over hour angles
    Ra_24_2 = np.sum(Ra_24_1, axis=0)

    Ra_24_temp = Ra_24_2 * 0.5 * dr * (1367 / 24)

    delta = et_common.delta_func(doy)
    omegas = et_common.omega_sunset_func(lat, delta)
    theta = (omegas * np.sin(lat) * np.sin(delta) +
             np.cos(lat) * np.cos(delta) * np.sin(omegas))

    Ra_24_min_flat = ((1367 * ra_min / math.pi) * et_common.dr_func(doy) * theta)
    Ra_24_min = Ra_24_min_flat / np.cos(slope)

    Ra_24 = np.where(Ra_24_temp > Ra_24_min_flat, Ra_24_temp / np.cos(slope), Ra_24_min)

    return Ra_24


def rl_in_func(tau, ts, ea_coef1=0.85, ea_coef2=0.09):
    """Incoming Longwave Radiation

    Parameters
    ----------
    tau : array_like
        Broadband atmospheric transmissivity.
    ts : array_like
        Surface temperature [K].
    ea_coef1 : float, optional
        Empirical coefficient for computing ea (the default is 0.85 per [1]_).
    ea_coef2 : float, optional
        Empirical coefficient for computing ea (the default is 0.09 per [1]_).

    Returns
    -------
    ndarray

    Notes
    -----
    ea = 0.85 * (-log(tau) ** 0.09)
    rl_in = 5.67E-8 * ea * (ts ** 4)

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    rl_in = np.copy(tau).astype(np.float64)
    np.log(rl_in, out=rl_in)
    np.negative(rl_in, out=rl_in)
    np.power(rl_in, ea_coef2, out=rl_in)
    rl_in *= (ea_coef1 * 5.67E-8)
    rl_in *= np.power(ts, 4)
    return rl_in.astype(np.float32)


def calculate_radiation_lw_incoming_mountain(
        surface_temperature_dem_cold_pixels,
        transmissivity,
        slope,
        broad_band_emissivity,
        surface_temperature_terrain):
    '''
    Computes the incoming longwave radiation incoming adjusting for terrain.
    Slope in degrees.

           .. topic:: References

       - "METRIC Applications Manual - Version 3.0"
         Allen R.G., Trezza R., Tasumi M., Kjaersgaard J., (2014)"""
    '''

    RLin = ne.evaluate(
        '(0.85*((-log(bb_trans))**0.09))*(5.67*(10**-8)*(ts_cold_dem**4))\
        *(0.75+0.25*cos(slope)-0.5*(slope)/pi)\
        +(bb_emiss*(5.67*(10**-8))*(lst_terrain**4))*(1-(0.75+0.25*cos(slope)-0.5*(slope)/pi))',
        {
            'bb_trans': transmissivity,
            'slope': slope * (math.pi / 180),
            'pi': math.pi,
            'bb_emiss': broad_band_emissivity,
            'lst_terrain': surface_temperature_terrain,
            'ts_cold_dem': surface_temperature_dem_cold_pixels
        }
    )

    return RLin


def rl_out_func(rl_in, ts, em_0):
    """Outgoing Longwave Radiation (Emitted + Reflected)

    Parameters
    ----------
    rl_in : array_like
        Incoming longwave radiation [W m-2].
    ts : array_like
        Surface temperature [K].
    em_0 : array_like
        Broadband surface emissivity.

    Returns
    -------
    ndarray

    Notes
    -----
    rl_out = 5.67E-8 * em_0 * (ts ** 4) + rl_in * (1 - em_0)

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    rl_out = np.copy(ts).astype(np.float64)
    np.power(rl_out, 4, out=rl_out)
    rl_out *= em_0
    rl_out *= 5.67E-8
    rl_out += rl_in
    rl_out -= em_0 * rl_in
    return rl_out.astype(np.float32)


def rso_instant_mountain_func(rso_inst_flat, lat, lon, slope, cos_theta, cos_theta_mountain, pair, ea, dr, doy, hour):

    """Instantaneous clear-sky solar radiation adjusted for slope and other terrain effects

       .. topic:: References

       - "METRIC Applications Manual - Version 3.0"
         Allen R.G., Trezza R., Tasumi M., Kjaersgaard J., (2014)

       - Allen, R. G., Trezza, R., & Tasumi, M. (2006).
         Analytical integrated functions for daily solar radiation on slopes.
         Agricultural and Forest Meteorology, 139(1-2), 55-73.

         """

    slope = slope * (math.pi / 180)

    sc = et_common.seasonal_correction_func(doy)
    img_time = et_common.solar_time_rad_func(lon, hour, sc)
    delta = et_common.delta_func(doy)

    w = et_common.precipitable_water_func(pair, ea)
    TauB = tau_direct_func(pair, w, cos_theta)
    TauD = tau_diffuse_func(TauB)

    pi = math.pi
    terrain_albedo = np.zeros_like(TauB) + 0.2

    Rso_inst = ne.evaluate(
        "TauB*1367*(1+0.033*cos(doy*2*pi/365))*cos_theta_mm\
        +TauD*1367*(1+0.033*cos(doy*2*pi/365))*cos(pi/2-arcsin(sin(lat)\
        *sin(delta)+cos(lat)*cos(delta)*cos(pi/12*(img_time-12))))\
        *(0.75+0.25*cos(slope)-0.5*(slope)/pi)\
        +rso_inst_flat*terrain_albedo*(1-(0.75+0.25*cos(slope)-0.5*(slope)/pi))",
        {'slope': slope,
         'lat': lat,
         'doy': doy,
         'rso_inst_flat': rso_inst_flat,
         'terrain_albedo': terrain_albedo,
         'img_time': img_time,
         'delta': delta,
         'TauB': TauB,
         'TauD': TauD,
         'pi': pi,
         'cos_theta_mm': cos_theta_mountain
         })

    return Rso_inst


def rs_in_func(cos_theta, tau, dr, gsc=1367.0):
    """Incoming Shortwave Radiation

    Parameters
    ----------
    cos_theta : array_like
    tau : array_like
    dr : float
    gsc : float, optional
        Solar constant [W m-2] (the default is 1367.0).

    Returns
    -------
    ndarray

    Notes
    -----
    rs_in = gsc * cos_theta * tau / dr

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    rs_in = np.copy(cos_theta).astype(np.float64)
    rs_in *= tau
    rs_in *= (gsc * dr)
    return rs_in.astype(np.float32)


def rs_out_func(rs_in, albedo_sur):
    """Outgoing Shortwave Radiation

    Parameters
    ----------
    rs_in : array_like
        Incoming shortwave radiation [W m-2].
    albedo_sur : array_like
        Surface albedo.

    Returns
    -------
    ndarray

    Notes
    -----
    rs_out = rs_in * albedo

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    rs_out = np.copy(rs_in).astype(np.float64)
    rs_out *= albedo_sur
    return rs_out.astype(np.float32)


def rn_func(rs_in, rs_out, rl_in, rl_out):
    """Net Radiation

    Parameters
    ----------
    rs_in : array_like
        Incoming shortwave radiation [W m-2].
    rs_out : array_like
        Outgoing shortwave radiation [W m-2].
    rl_in : array_like
        Incoming longwave radiation [W m-2].
    rl_out : array_like
        Outgoing longwave radiation [W m-2].

    Returns
    -------
    ndarray

    Notes
    -----
    rn = rs_in - rs_out + rl_in - rl_out

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    rn = np.copy(rs_in)
    rn -= rs_out
    rn += rl_in
    rn -= rl_out
    return rn


def rn_24_func(albedo_sur, rs_in, lat, doy, cs=110):
    """Daily Net Radiation

    Parameters
    ----------	
    albedo_sur : array_like
        Surface albedo.
    rs_in : array_like
        Incoming shortwave radiation [W m-2]
    lat : array_like
        Latitude [rad].
    doy : int
        Day of year.
    cs : float
        Slob calibration coefficient (the default is 110 W m-2 [1]_).

    Returns
    -------	
    ndarray

    Notes
    -----
    This function is calling the et_common.ra_daily_func() but could be
    changed to use the refet.calcs._ra_daily() function instead.

    rnl_24 = cs * (rs_in / ra)
    rn_24 = (1 - albedo_sur) * rs_in - rnl_24

    References
    ----------
    .. [1] de Bruin, H.A.R. (1987). From Penman to Makkink. Proceedings and
       Information: TNO Committee on Hydrological Research No. 39,
       J. C. Hooghart, Ed., Netherlands Organization for Applied Scientific
       Research, 5-30.
    .. [2] de Bruin and Stricker (2000).
    .. [3] Bastiaanssen, W., Noordman, E., Pelgrum, H., Davids, G., Thoreson, B.,
       Allen, R. (2005). SEBAL model with remotely sensed data to improve
       water-resources management under actual field conditions.
       Journal of Irrigation and Drainage Engineering, 131(1).
       https://doi.org/10.1061/(ASCE)0733-9437(2005)131:1(85)
    .. [4] de Bruin, H.A.R, Trigo, I.F., Bosveld, F.C., & Meirink, J.F. (2016).
       A thermodynamically based model for actual evapotranspiration of an
       extensive grass field close to FAO reference, suitable for remote
       sensing application. Journal of Hydrometeorology 17.
       https://doi.org/10.1175/JHM-D-15-0006.1

    """
    # Net longwave radiation at the cold and hot calibration points
    rnl_24 = et_common.ra_daily_func(lat=lat, doy=doy)
    np.reciprocal(rnl_24, out=rnl_24)
    rnl_24 *= rs_in
    rnl_24 *= cs

    rn_24 = 1 - albedo_sur
    rn_24 *= rs_in
    rn_24 -= rnl_24
    return rn_24


def rn_mountain_func(Rso_inst_flat, Rso_inst, RLin, bb_emiss, ts, albedo):

    """Instantaneous Net Radiation

       .. topic:: References

       - "METRIC Applications Manual - Version 3.0"
         Allen R.G., Trezza R., Tasumi M., Kjaersgaard J., (2014)"""

    Rso_adj = np.zeros_like(albedo)

    rn = ne.evaluate(
        '((1-albedo)*Rso_inst+RLin-(bb_emiss*(5.67*10**-8)*(lst**4))-((1-bb_emiss)*RLin))', {
            'Rso_inst': Rso_inst,
            'albedo': albedo,
            'RLin': RLin,
            'bb_emiss': bb_emiss,
            'lst': ts
        })

    rn_1 = ne.evaluate(
                   '((1-albedo)*(Rso_inst**(1.0/2.0))*(Rso_inst_flat**(1.0/2.0))+RLin\
                   -(bb_emiss*(5.67*10**-8)*(lst**4))-((1-bb_emiss)*RLin))', {
                       'Rso_inst': Rso_inst,
                       'albedo': albedo,
                       'Rso_inst_flat': Rso_inst_flat,
                       'RLin': RLin,
                       'bb_emiss': bb_emiss,
                       'lst': ts
                   })
    lim_1 = Rso_inst / Rso_inst_flat

    rn_2 = ne.evaluate(
                   '((1-albedo)*Rso_inst+RLin-(bb_emiss*(5.67*10**-8)*(lst**4))-((1-bb_emiss)*RLin))', {
                       'Rso_inst': Rso_inst,
                       'albedo': albedo,
                       'RLin': RLin,
                       'bb_emiss': bb_emiss,
                       'lst': ts
                   })

    rn = np.where(np.logical_and(lim_1 <= 0, Rso_adj == 1.0), rn_1, rn)
    rn = np.where(lim_1 >= 1, rn_2, rn)

    return rn


def rn_24_slob_func(lat, ts, ts_cold, ts_hot, lapsed_ts_cold, lapse, ts_dem_dry, ts_dem_point_array, albedo_sur,
                    rso_daily, ra_daily, doy, cold_xy, hot_xy, radiation_longwave_in_flat,
                    radiation_longwave_in_mountain, radiation_longwave_out):
    """Daily Net Radiation - Slob method

       .. topic:: References

       - "METRIC Applications Manual - Version 3.0"
         Allen R.G., Trezza R., Tasumi M., Kjaersgaard J., (2014)"""

    ts_dem_hot = ts_dem_point_array[1]
    num_daylight_hrs = daylight_hours_func(lat, doy)
    del lat

    # (rso_daily / ra_daily) is equivalent to daily Tau = TauB+TauD
    tau = rso_daily / ra_daily
    del ra_daily

    # 0.5 is the daylength weight
    rnl_cold_24 = 140.0 * 0.5 + (0.5 * 180) * tau
    rnl_hot_24 = 115.0 * 0.5 + (0.5 * 145) * tau

    ts_dry = ts_hot + (ts_dem_dry - ts_dem_hot)
    lapsed_ts_dry = ts_dry - lapse
    del ts_dry, lapse, ts_dem_hot

    net_radiation_mountain_reduction = ne.evaluate(
        '(rl_out-rl_in_mtn)/(rl_out-rl_in)',
        {
            'rl_out': radiation_longwave_out,
            'rl_in_mtn': radiation_longwave_in_mountain,
            'rl_in': radiation_longwave_in_flat
        }
    )
    del radiation_longwave_in_flat, radiation_longwave_in_mountain, radiation_longwave_out

    net_radiation_mountain_reduction = np.maximum(np.minimum(net_radiation_mountain_reduction, 1.0), 0.6)

    # TODO: Is it necessary to extract values at pixels for Rnl_hot / Rnl_cold?
    # Rnl_hot = np.array(drigo.array_value_at_xy(input_array, input_geo, input_xy,
    #                   input_nodata=None, band=1)()
    #     .cell_value_set(
    #         Rnl_hot, 'rnl_hot', cold_xy, hot_xy))
    #
    #     # This creates a variable from the temp of hot and cold pixels
    #     # (for Ts correction)
    #     cold_px_temp = ts_array.item(0)
    #     hot_px_temp = ts_array.item(1)

    rnl_24_pixel = (
        ((ts - lapsed_ts_cold) / (lapsed_ts_dry - lapsed_ts_cold)) *
        (rnl_hot_24 - rnl_cold_24) + rnl_cold_24)
    rnl_24_pixel = np.where(rnl_24_pixel < rnl_cold_24, rnl_cold_24, rnl_24_pixel)
    rnl_24_pixel = np.where(rnl_24_pixel > rnl_hot_24, rnl_hot_24, rnl_24_pixel)

    del rnl_cold_24, rnl_hot_24
    del ts, lapsed_ts_cold, lapsed_ts_dry

    rn_daily = ne.evaluate(
        '(1 - albedo_tall_vegetation) * rso_daily * 24.0\
        / (effective_daylight_weight * 24.0 + (1 - effective_daylight_weight) * num_daylight_hrs)\
        - rnl * radiation_reduction',
        {
            'rso_daily': rso_daily,
            'albedo_tall_vegetation': albedo_sur,
            'effective_daylight_weight': 0.5,
            'num_daylight_hrs': num_daylight_hrs,
            'rnl': rnl_24_pixel,
            'radiation_reduction': net_radiation_mountain_reduction
        }
    )

    return rn_daily


def g_ag_func(lai, ts, rn, coef1=1.80, coef2=0.084):
    """Calculate ground heat flux for agriculture using METRIC approach

    Parameters
    ----------
    lai : array_like
        Leaf area index.
    ts : array_like
        Surface temperature [K].
    rn : array_like
        Net radiation [W m-2].
    coef1 : float
        Coefficient (the default is 1.80).
    coef2 : float
        Coefficient (the default is 0.084).

    Returns
    -------
    ndarray

    Notes
    -----
    Coef1 and coef2 are exposed in order to apply a custom G function.

    g = np.where(
        lai_toa >= 0.5,
        (0.05 + (0.18 * exp(-0.521 * lai))) * rn,
        coef1 * (ts - 273.16) + (coef2 * rn))

    References
    ----------
    .. [1] Tasumi (2003)
    .. [2] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    a = np.copy(lai).astype(np.float64)
    a *= -0.521
    np.exp(a, out=a)
    a *= 0.18
    a += 0.05
    a *= rn
    b = ts - 273.16
    b *= coef1
    b /= rn
    b += coef2
    b *= rn
    return np.where(lai >= 0.5, a, b).astype(np.float32)


def g_sebal_func(ts, albedo_sur, ndvi):
    """Calculate ground heat flux using SEBAL approach

    Parameters
    ----------
    ts : array_like
        Surface temperature [K].
    albedo_sur : array_like
        Surface albedo.
    ndvi : array_like
        Normalized difference vegetation index.

    Returns
    -------
    ndarray

    Notes
    -----
    In [1]_, ts is listed as "radiometric surface temperature".

    g = (ts - 273.15) * (0.0038 + 0.0074 * albedo) * (1 - 0.98 * ndvi ** 4)

    References
    ----------
    .. [1] Bastiaanssen, W. (2000). SEBAL-based sensible and latent heat fluxes
       in the irrigated Gediz Basin, Turkey. Journal of Hydrology, 229(1-2).
       https://doi.org/10.1016/S0022-1694(99)00202-4
    .. [2] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    g = np.copy(ndvi).astype(np.float64)
    np.power(g, 4, out=g)
    g *= -0.98
    g += 1
    g *= ts
    g *= (albedo_sur * 0.0074 + 0.0038)
    return g


def zom_func(lai, landuse, zom_remap):
    """Generate Zom (roughness) values based on the landuse type

    Parameters
    ----------
    lai : ndarray
        Leaf area index.
    landuse : ndarray
        Landuse.
    zom_remap : dict
        Mapping of landuse types to zom values in JSON format with key/value
        both string type (i.e. "11" : "0.005").

    Returns
    -------
    ndarray

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    zom = np.full(lai.shape, 0.005, dtype=np.float32)

    for lu_code in np.unique(landuse):
        # What should be the default zom value?
        # Convert the landuse array values to strings for now.
        try:
            lu_value = zom_remap[str(lu_code)]
        except:
            lu_value = 'lai'

        if lu_value.lower() == 'perrier':
            zom[landuse == lu_code] = perrier_zom_func(lai[landuse == lu_code])
        elif lu_value.lower() == 'lai':
            zom[landuse == lu_code] = np.maximum(
                lai[landuse == lu_code] * 0.018, 0.005)
        else:
            zom[landuse == lu_code] = float(lu_value)

    zom[np.isnan(lai)] = np.nan
    return zom


def perrier_zom_func(lai):
    """Perrier Zom

    Parameters
    ----------
    lai : ndarray
        Leaf area index.

    Returns
    -------
    ndarray

    Notes
    -----
    Minimum zom is 0.005 m equal to bare soil. Dec 28 09, JK
    The use of the function is applicable for tall vegetation (forests).
    The canopy distribution coefficient, a, is assumed to be a=0.6,
        i.e. slightly top heavy canopy.
    The vegetation height is estimated as h=2.5LAI (LAImax=6 -> 2.5*6=15 m),
        compared to h=0.15LAI for agriculture crops.

    References
    ----------
    .. [1] Perrier, A. (1982). Land surface processes: Vegetation.
       In Land Surface Processes in Atmospheric General Circulation Models;
       Eagelson, P.S., Ed.; Cambridge University Press: Cambridge, UK;
       pp. 395-448.
    .. [2] Allen, R., Irmak, A., Trezza, R., Hendrickx, J., Bastiaanssen, W.,
       & Kjaersgaard, J. (2011). Satellite-based ET estimation in agriculture
       using SEBAL and METRIC. Hydrologic Processes, 25, 4011-4027.
       https://doi.org/10.1002/hyp.8408
    .. [3] Santos (2012)

    """
    perrier = -1.2 * lai
    perrier /= 2.
    np.exp(perrier, out=perrier)
    perrier = ((1 - perrier) * perrier) * (2.5 * lai)
    return np.maximum(perrier, 0.005, dtype=np.float32)


# The following equations are float specific, separate from equations below.
# This is indicated by having "calibration" in the function name.
def le_calibration_func(etr, kc, ts):
    """Latent heat flux at the calibration points

    Parameters
    ----------
    etr : scalar or array_like
    kc : scalar or array_like
    ts : scalar or array_like
        Surface temperature [K].

    Returns
    -------
    scalar or array_like

    Notes
    -----
    1000000 / 3600 in [1] was simplified to 2500 / 9

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    return etr * kc * (2.501 - 2.361E-3 * (ts - 273)) * 2500 / 9


def dt_calibration_func(h, rah, density):
    """

    Parameters
    ----------
    h : scalar or array_like
        Sensible heat flux [W m-3].
    rah : scalar or array_like
        Aerodynamic resistance to heat transport [s m-1].
    density : scalar or array_like
        Air density [kg m-3].

    Returns
    -------
    scalar or array_like

    Notes
    -----
    The 1004.0 term is the specific heat capacity of air [J kg-1 K-1].

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    return (h * rah) / (density * 1004.)


def l_calibration_func(h, air_density, u_star, ts):
    """

    Parameters
    ----------
    h : scalar or array_like
        Sensible heat flux [W m-3].
    air_density : scalar or array_like
        Air density [kg m-3].
    u_star : scalar or array_like
        Friction velocity [m s-1].
    ts : scalar or array_like
        Surface temperature [K].

    Returns
    -------
    scalar or array_like

    Notes
    -----
    Return -1000 if h is zero to avoid dividing by zero.

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    return np.where(
        h != 0,
        ((-1004. * air_density * (u_star ** 3.0) * ts) / (0.41 * 9.81 * h)),
        -1000)


def h_func(air_density, dt, rah):
    """Sensible Heat Flux [W/m^2]

    Parameters
    ----------
    air_density : array_like
        Air density [kg m-3].
    dt : array_like
        Near surface temperature difference [K].
    rah : array_like
        Aerodynamic resistance to heat transport [s m-1].

    Returns
    -------
    ndarray

    Notes
    -----
    h = air_density * 1004 * dt / rah

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    h = np.array(air_density, copy=True, ndmin=1)
    h *= 1004.
    h *= dt
    h /= rah
    return h


def u_star_func(u3, z3, zom, psi_z3, wind_coef=1):
    """

    Parameters
    ----------
    u3 : array_like

    z3 : array_like

    zom : array_like

    psi_z3 : array_like

    wind_coef : float, optional
        (the default is 1).

    Returns
    -------
    u_star : ndarray
        Friction velocity [m s-1].

    Notes
    -----
    u_star = (u3 * wind_coef * 0.41) / (log(z3 / zom) - psi_z3)

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    u_star = np.array(zom, copy=True, ndmin=1)
    np.reciprocal(u_star, out=u_star)
    u_star *= z3
    oldsettings = np.geterr()
    np.seterr(invalid='ignore')
    np.log(u_star, out=u_star)
    np.seterr(invalid=oldsettings['invalid'])
    u_star -= psi_z3
    np.reciprocal(u_star, out=u_star)
    u_star *= (u3 * wind_coef * 0.41)
    return u_star


def rah_func(z_flt_dict, psi_z2, psi_z1, u_star, excess_res=0):
    """"""
    rah = np.array(psi_z1, copy=True, ndmin=1)
    rah -= psi_z2
    rah += math.log(z_flt_dict[2] / z_flt_dict[1])
    rah /= 0.41
    rah /= u_star
    rah += excess_res
    return rah


def density_func(elev, ts, dt):
    """

    Parameters
    ----------
    elev : array_like
        Elevation [m].
    ts : array_like
        Surface temperature [K].
    dt : array_like
        Near surface temperature difference [K].

    Returns
    -------
    air_density : ndarray
        Air density [kg m-3].

    Notes
    -----
    den = (1000. * 101.3 * (((293.15 - 0.0065 * elev) / 293.15) ** 5.26) /
           (1.01 * (ts - dt) * 287))

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    air_density = np.array(elev, copy=True, ndmin=1).astype(np.float64)
    air_density *= -0.0065
    air_density += 293.15
    air_density /= 293.15
    np.power(air_density, 5.26, out=air_density)
    air_density *= ((1000 * 101.3) / (1.01 * 287))
    air_density /= (ts - dt)
    return air_density.astype(np.float32)


def x_func(l, z):
    """

    Parameters
    ----------
    l : array_like

    z : array_like


    Returns
    -------
    ndarray

    Notes
    -----
    x = np.where(l < 0, power((1 - 16 * z / l), 0.25), 0)

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    x = np.array(l, copy=True, ndmin=1)
    l_mask = (x > 0)
    np.reciprocal(x, out=x)
    x *= (-16 * z)
    x += 1
    np.power(x, 0.25, out=x)
    x[l_mask] = 0
    del l_mask
    return x


def psi_func(l, z_index, z):
    """

    Parameters
    ----------
    l : array_like

    z_index : int

    z : array_like


    Returns
    -------
    ndarray

    Notes
    -----
    psi(3) = np.where(
        l > 0,
        (-5 * 2 / l),
        ((2 * log((1 + x) / 2)) + log((1 + (x ** 2)) / 2) - (2 * atan(x)) + (pi / 2)))
    psi = np.where(l > 0, (-5 * z / l), (2 * log((1 + (x ** 2)) / 2.)))

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    # Begin calculation of Psi unstable
    x = x_func(l, z)
    psi = np.array(x, copy=True, ndmin=1)
    np.power(x, 2, out=psi)
    psi += 1
    psi /= 2.
    oldsettings = np.geterr()
    np.seterr(invalid='ignore')
    np.log(psi, out=psi)
    np.seterr(invalid=oldsettings['invalid'])

    # Adjust Psi unstable calc based on height
    if z_index == 3:
        psi_temp = np.copy(x)
        psi_temp += 1
        psi_temp /= 2.
        oldsettings = np.geterr()
        np.seterr(invalid='ignore')
        np.log(psi_temp, out=psi_temp)
        np.seterr(invalid=oldsettings['invalid'])
        psi_temp *= 2.
        psi += psi_temp
        del psi_temp
        psi_temp = np.copy(x)
        np.arctan(x, out=psi_temp)
        psi_temp *= 2.
        psi -= psi_temp
        del psi_temp
        psi += (0.5 * math.pi)
    else:
        psi *= 2.

    del x

    # Calculate Psi stable for all pixels
    psi_stable = np.array(l, copy=True, ndmin=1)
    np.reciprocal(psi_stable, out=psi_stable)
    if z_index == 3:
        psi_stable *= (-5 * 2)
    else:
        psi_stable *= (-5 * z)

    # Only keep Psi stable for pixels with l > 0
    l_mask = np.array(l, copy=True, ndmin=1) > 0
    psi[l_mask] = psi_stable[l_mask]

    return psi


def dt_func(dt_adjust_flag, ts_dem, a, b, ts_cold_threshold, ts_hot_threshold, dt_slope_factor=8):
    """dT function
    Parameters
    ----------
    ts_dem : array_like
        Surface temperature [K].  As described in [1], this should be the
        delapsed surface temperature.
    a : float
        Calibration parameter.
    b : float
        Calibration parameter.
    ts_cold_threshold: float
        Surface temperature [K] threshold below cold pixel
    ts_hot_threshold: float
        Surface temperature [K] threshold above hot pixel
    dt_slope_factor: float
        Factor reducing the slope (b) of the dT function above and below the hot and cold pixel

    Returns
    -------
    ndarray

    Notes
    -----
    dt = a * ts + b

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)
    .. [2] "METRIC Applications Manual - Version 3.0"
         Allen R.G., Trezza R., Tasumi M., Kjaersgaard J., (2014)"""
    """"""
    dt = np.copy(ts_dem)
    dt *= a
    dt += b
    if dt_adjust_flag:
        dt_adjust = ts_dem - ts_hot_threshold
        dt_adjust *= (a / dt_slope_factor)
        dt_adjust += (a * ts_hot_threshold + b)
        dt_adjust_low = ts_dem - ts_cold_threshold
        dt_adjust_low *= (a / dt_slope_factor)
        dt_adjust_low += (a * ts_cold_threshold + b)
        np.where(ts_dem > ts_hot_threshold, dt_adjust, dt)
        np.where(ts_dem < ts_cold_threshold, dt_adjust_low, dt)

    return dt


def l_func(dt, u_star, ts, rah):
    """

    Parameters
    ----------
    dt : array_like
        Near surface temperature difference [K].
    u_star : array_like
        Friction velocity [m s-1].
    ts : array_like
        Surface temperature [K].
    rah : array_like
        Aerodynamic resistance to heat transport [s m-1].

    Returns
    -------
    l : ndarray

    Notes
    -----
    dt_mod = np.where((np.absolute(dt)==0.), -1000., dt)

    l = -((u_star ** 3) * ts * rah) / (0.41 * 9.81 * dt_mod)

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    # Change zeros to -1000 to avoid divide by zero
    dt[dt == 0] = -1000
    l = np.power(u_star, 3)
    l *= ts
    l *= rah
    l /= -(0.41 * 9.81)
    l /= dt
    return l



def le_func(rn, g, h):
    """Latent Heat Flux [W/m^2]

    Parameters
    ----------
    rn : array_like
        Net radiation [W m-2].
    g : array_like
        Ground heat flux [W m-2].
    h : array_like
        Sensible heat flux into the air [W m-2]

    Returns
    -------
    ndarray

    Notes
    -----
    le = rn - g - h

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    le = np.copy(rn)
    le -= g
    le -= h
    return le


def ef_func(le, rn, g):
    """Evaporative fraction

    Parameters
    ----------
    le : array_like
        Latent heat flux [W m-2].
    rn : array_like
        Net radiation [W m-2].
    g : array_like
        Ground heat flux [W m-2].

    Returns
    -------
    ndarray

    Notes
    -----
    ef = le / (rn - g)

    References
    ----------
    .. [1] Bastiaanssen, W., Noordman, E., Pelgrum, H., Davids, G., Thoreson, B.,
       Allen, R. (2005). SEBAL model with remotely sensed data to improve
       water-resources management under actual field conditions.
       Journal of Irrigation and Drainage Engineering, 131(1).
       https://doi.org/10.1061/(ASCE)0733-9437(2005)131:1(85)
    .. [2] Allen, R., Irmak, A., Trezza, R., Hendrickx, J., Bastiaanssen, W.,
       & Kjaersgaard, J. (2011). Satellite-based ET estimation in agriculture
       using SEBAL and METRIC. Hydrologic Processes, 25, 4011-4027.
       https://doi.org/10.1002/hyp.8408

    """
    ef = np.copy(rn)
    ef -= g
    np.reciprocal(ef, out=ef)
    ef *= le
    return ef


def heat_vaporization_func(ts):
    """Latent heat of vaporization [J kg-1]

    Parameters
    ----------
    ts : array_like
        Surface temperature [K].

    Returns
    -------
    ndarray

    Notes
    -----
    lambda = (2.501 - 0.00236 * (ts - 273.15)) * 1E6

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    heat_vaporization = np.copy(ts).astype(np.float64)
    heat_vaporization -= 273.15
    heat_vaporization *= -0.00236
    heat_vaporization += 2.501
    heat_vaporization *= 1E6
    return heat_vaporization.astype(np.float32)


def et_inst_func(le, ts):
    """ET instantaneous [mm/hr]

    Parameters
    ----------
    le : array_like
        Latent heat flux [W m-2].
    ts : array_like
        Surface temperature [K].

    Returns
    -------
    ndarray

    Notes
    -----
    et_inst = 3600 * le / heat_vaporization

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    et_inst = np.copy(le).astype(np.float64)
    et_inst *= 3600
    et_inst /= heat_vaporization_func(ts)
    return et_inst.astype(np.float32)


def etrf_func(et_inst, etr):
    """ET Reference Fraction - ETrF

    Parameters
    ----------
    et_inst : array_like
        ET at time of overpass [mm hr-1].
    etr : array_like
        Reference ET at time of overpass [mm hr-1].

    Returns
    -------
    array_like

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    return et_inst / etr


def et_24_func(etr_24hr, etrf):
    """ET 24hr [mm/day]

    Parameters
    ----------
    etr_24hr : array_like
        Daily reference ET [mm].
    etrf : array_like
        Fraction of reference ET (ETrF).

    Returns
    -------
    array_like

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    return etr_24hr * etrf


def calculate_et_ef(lat, doy, rn_daily, rn, g, et_inst):
    """Compute the ET using the evaporative fraction.

    .. topic:: References

        - "METRIC Applications Manual - Version 3.0"
          Allen R.G., Trezza R., Tasumi M., Kjaersgaard J., (2014)

    """

    effective_daylight_weight = 0.5
    num_daylight_hrs = daylight_hours_func(lat, doy)

    # Allen et al. (2014)
    et24_non_ag = ne.evaluate(
        'et_inst * (num_daylight_hrs * (1.0 - effective_daylight_weight) + 24.0 * effective_daylight_weight)\
         / (rn - g) * (rn_daily - g * (1.0 - effective_daylight_weight))',
        {
            'et_inst': et_inst,
            'num_daylight_hrs': num_daylight_hrs,
            'effective_daylight_weight': effective_daylight_weight,
            'rn': rn,
            'g': g,
            'rn_daily': rn_daily
        }
    )

    return et24_non_ag
