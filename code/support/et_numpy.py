#--------------------------------
# Name:         et_numpy.py
# Purpose:      NumPy ET functions
#--------------------------------

# import logging
import math

import numpy as np

import et_common


def cos_theta_spatial_func(time, doy, dr, lon, lat):
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
    omega = et_common.omega_func(et_common.solar_time_rad_func(time, lon, sc))
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
    sc = et_common.seasonal_correction_func(doy)
    delta = et_common.delta_func(doy)
    omega = et_common.omega_func(et_common.solar_time_rad_func(time, lon, sc))
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


def ts_lapsed_func(ts, elevation, datum, lapse_rate=6.0):
    """Lapse surface temperature based on elevation

    Parameters
    ----------
    ts : array_like
        Surface temperature [K].
    elevation : array_like
        Elevation [m].
    datum : float
    lapse_rate : float

    Returns
    -------
    ndarray

    Notes
    -----


    References
    ----------


    """
    ts_adjust = np.copy(elevation).astype(np.float64)
    ts_adjust -= datum
    ts_adjust *= (lapse_rate * -0.001)
    ts_adjust += ts
    return ts_adjust.astype(np.float32)


def ts_delapsed_func(ts, elevation, datum, lapse_rate=6.0):
    """Delapse surface temperature based on elevation

    Parameters
    ----------
    ts : array_like
        Surface temperature [K].
    elevation : array_like
        Elevation [m].
    datum : float
    lapse_rate : float

    Returns
    -------
    ndarray

    Notes
    -----


    References
    ----------


    """
    ts_adjust = np.copy(elevation).astype(np.float64)
    ts_adjust -= datum
    ts_adjust *= (lapse_rate * 0.001)
    ts_adjust += ts
    return ts_adjust.astype(np.float32)


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


def rah_func(z_flt_dict, psi_z2, psi_z1, u_star):
    """

    Parameters
    ----------
    z_flt_dict : dict

    psi_z2 : array_like

    psi_z1 : array_like

    u_star : array_like
        Friction velocity [m s-1].

    Returns
    -------
    rah : ndarray
        Aerodynamic resistance to heat transport [s m-1].

    Notes
    -----
    rah = ((log(z2 / z1) - psi_z2 + psi_z1) / (0.41 * u_star))

    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)

    """
    rah = np.array(psi_z1, copy=True, ndmin=1)
    rah -= psi_z2
    rah += math.log(z_flt_dict[2] / z_flt_dict[1])
    rah /= 0.41
    rah /= u_star
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
    # return np.where((l > 0), psi_stable, psi_unstable)


# The following equations are array specific and are separate from the
# "calibration" functions above
def dt_func(ts, a, b):
    """

    Parameters
    ----------
    ts : array_like
        Surface temperature [K].  As described in [1]_, this should be the
        delapsed surface temperature.
    a : float
        Calibration parameter.
    b : float
        Calibration parameter.

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

    """
    dt = np.copy(ts)
    dt *= a
    dt += b
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
