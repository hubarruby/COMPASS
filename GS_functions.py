#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
from numpy.linalg import norm
import pandas as pd
#igrf stuff:
import igrf_utils as iut 
from scipy import interpolate
#rotations:
from scipy.spatial.transform import Rotation as rot
#camera stuff:
import cv2 as cv
#getting sun az/el:
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from sunpy.coordinates import frames, sun


#function to find the start of the data ('O' is used because that character indicates the first 'ORI' label)
def find_data_start(txt_file_name):
    with open (txt_file_name, 'r') as f:
        for line_count, line in enumerate(f):
            for count, char in enumerate(line):
                if (count == 0) & (char == 'O'):
                    return int(line_count)

#function to turn strings into floats, or keeps them as strings
def num_or_string_parse(s):
    try:
        return float(s)
    except ValueError:
        return s
    
#function to help with getting IGRF info
#Note: This code was modified/trimmed from the original IGRF code so that we could use it in COMPASS.
#To see the original code, clikc on "python 3.7 package" at https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
def option1(ltd,lnd,alt,date):
    latd = iut.check_float(ltd)
    lond = iut.check_float(lnd)

    lat, lon = iut.check_lat_lon_bounds(latd,0,lond,0)
    colat = 90-lat

    itype = 1
    alt = iut.check_float(alt)
    alt, colat, sd, cd = iut.gg_to_geo(alt, colat)

    date = iut.check_float(date)

    return date, alt, lat, colat, lon, itype, sd, cd

# 2nd function to help with getting IGRF info
#returns igrf xyz, and declination (for converting from magnetic north to true north later on)
def get_igrf_readings(ltd,lnd,alt,date):
    # Load in the file of coefficients
    IGRF_FILE = r'./IGRF13.shc'
    igrf = iut.load_shcfile(IGRF_FILE, None)
    date, alt, ltd, colat, lnd, itype, sd, cd = option1(ltd,lnd,alt,date)
    # Interpolate the geomagnetic coefficients to the desired date(s)
    f = interpolate.interp1d(igrf.time, igrf.coeffs, fill_value='extrapolate')
    coeffs = f(date)    
    # Compute the main field B_r, B_theta and B_phi value for the location(s) 
    Br, Bt, Bp = iut.synth_values(coeffs.T, alt, colat, lnd,
                                  igrf.parameters['nmax'])
    epoch = (date-1900)//5 ; epoch_start = epoch*5
    # Add 1900 back on plus 1 year to account for SV in nT per year (nT/yr):
    coeffs_sv = f(1900+epoch_start+1) - f(1900+epoch_start)   
    Brs, Bts, Bps = iut.synth_values(coeffs_sv.T, alt, colat, lnd,
                                     igrf.parameters['nmax'])
    # Use the main field coefficients from the start of each five epoch
    # to compute the SV for Dec, Inc, Hor and Total Field (F) 
    # [Note: these are non-linear components of X, Y and Z so treat separately]
    coeffsm = f(1900+epoch_start);
    Brm, Btm, Bpm = iut.synth_values(coeffsm.T, alt, colat, lnd,
                                     igrf.parameters['nmax'])
    X = -Bt; Y = Bp; Z = -Br # Rearrange to X, Y, Z components 
    dX = -Bts; dY = Bps; dZ = -Brs; Xm = -Btm; Ym = Bpm; Zm = -Brm
    # Rotate back to geodetic coords
    t = X; X = X*cd + Z*sd;  Z = Z*cd - t*sd
    t = dX; dX = dX*cd + dZ*sd;  dZ = dZ*cd - t*sd
    t = Xm; Xm = Xm*cd + Zm*sd;  Zm = Zm*cd - t*sd
    dec, hoz, inc, eff = iut.xyz2dhif(X,Y,Z)     # Compute the four non-linear components 
    decs, hozs, incs, effs = iut.xyz2dhif_sv(Xm, Ym, Zm, dX, dY, dZ)

    igrf_xyz = [Y, Z, X]
    return igrf_xyz, dec

#This function finds the vector direction of the magnetometer relative to north
#uses igrf and mag values, along with accelerometer and expected accelerometer values
#Outputs the rotation of a vector in the boom frame into north frame (used in the data comparison section)
#also outputs the "goodness of fit" of the rotation; if this is bigger than 0, then the vectors didn't 
#...align perfectly. As such, it is good to check this goodness of fit before trusting a rotation
def get_rot_v(mag_xyz, igrf_xyz, accel_xyz):
    ACCEL_DEFAULT = [0,1,0]
    mag_xyz = [i/norm(mag_xyz) for i in mag_xyz]
    igrf_xyz = [i/norm(igrf_xyz) for i in igrf_xyz]
    accel_xyz = [i/norm(accel_xyz) for i in accel_xyz]
    #find the rotation of a vector in the boom frame to the earth frame (north)
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html#scipy.spatial.transform.Rotation.align_vectors
    rot_v = rot.align_vectors([igrf_xyz, ACCEL_DEFAULT],[mag_xyz, accel_xyz])
    return rot_v[0], rot_v[1]

#this function uses the rotation vector from get_rot_v and finds the xyz coordinates
# of the boom in the earth frame
def get_boom_north_frame_xyz(rot_v):
    MAG_POINT_VECT = [0,0,1]
    mag_relative_to_north = rot_v.apply(MAG_POINT_VECT)
    return mag_relative_to_north

#converting a camera pixel coordinate from xyz to az/el, where +x = East, +y = Down, +z = North
#input an array with 3 values
#Output is in DEGREES, with the elevation negative (since our y axis is upsidedown)
def xyz_to_el_az(xyz_vect):
    rad_el = np.arctan2(-xyz_vect[1], np.sqrt(xyz_vect[0]**2 + xyz_vect[2]**2))
    rad_az = np.arctan2(xyz_vect[0], xyz_vect[2])
    el = rad_el/2/np.pi*360
    az = rad_az/2/np.pi*360
    return [el, az]

#converting an x,y pixel coordinate (where (0,0) is at the top left corner) to an xyz pointing direction...
#...using the inverse of the camera matrix; FOR CAM0
#outputs an array fo [x,y,z] which indicates the pointing direction of the pixel
def cam0_pix_to_xyz(xy_array, new_mtx0):
    return np.dot(np.linalg.inv(new_mtx0),[xy_array[0],xy_array[1],1])

#converting an x,y pixel coordinate (where (0,0) is at the top left corner) to an xyz pointing direction...
#...using the inverse of the camera matrix; FOR CAM1
def cam1_pix_to_xyz(xy_array, new_mtx1):
    return np.dot(np.linalg.inv(new_mtx1), [xy_array[0],xy_array[1],1])

#find center of sun coordinates for right and left image
#to increase sensitivity, decrease param2. Any lower than 8(ish) causes it to detect non-existent circles
def sun_centers(image_left, image_right):
    gray_left = cv.cvtColor(image_left, cv.COLOR_BGR2GRAY); blurry_face_left = cv.blur(gray_left, [5,5],0)
    circles_left = cv.HoughCircles(image = blurry_face_left, method = cv.HOUGH_GRADIENT, dp = 0.9, 
                                   minDist = 600, param1 = 50, param2 = 10, maxRadius = 50)
    try: circles_left = np.uint16(np.around(circles_left))
    except TypeError:
        xy_pix_coord_left = [float('NaN'), float('NaN')]
    else:
        xy_pix_coord_left = [circles_left[0][0][0], circles_left[0][0][1]]
    gray_right = cv.cvtColor(image_right, cv.COLOR_BGR2GRAY); blurry_face_right = cv.blur(gray_right, [5,5],0)
    circles_right = cv.HoughCircles(image = blurry_face_right, method = cv.HOUGH_GRADIENT,
                                    dp = 0.9, minDist = 600, param1 = 50, param2 = 10, maxRadius = 50)
    try:    circles_right = np.uint16(np.around(circles_right))
    except TypeError:
        xy_pix_coord_right = [float('NaN'), float('NaN')]
    else:
        xy_pix_coord_right = [circles_right[0][0][0], circles_right[0][0][1]]
    return [xy_pix_coord_left, xy_pix_coord_right]

#random potentially useful function (this is not used in Ground Software 2.1)
#converts any pixel value to a (0,0) centered point, for either the left image or the right image:
def get_zerocenter_coord(xy_pix_coord, half_image):
    h,w = half_image.shape[:2]
    center = [int(np.round(w/2)), int(np.round(h/2))]
    xy_zerocenter_coord = [xy_pix_coord[0] - center[0], xy_pix_coord[1] - center[1]]
    return xy_zerocenter_coord

#this function uses the calibration of cam0 to turn an xyz sun pos vector into a calibrated vector
#use once the cam0 sun position has been transformed into the boom frame, before transforming into earth frame
def calibrate_cam0_xyz(xyz_array):
    #something to account for the measured calibration curve happens here, redefining the xyz variables
    #currently we don't know what is going on for camera calibration, so just does nothing for now
    return xyz_array

#this function uses the calibration of cam1 to turn an xyz sun pos vector into a calibrated vector
# use once the cam1 sun position has been transformed into the boom frame, before transforming into earth frame
def calibrate_cam1_xyz(xyz_array):
    #something to account for the tested calibration curve happens here, redefining the xyz variables
    #currently we don't know what is going on for camera calibration, so just does nothing for now
    return xyz_array

#this is for getting the actual azimuth and elevation of the sun (in terms of true north)
#gps_time for input should be in the format '2013-09-21 16:00:00.0034'
#time input is UTC time
#not sure if altitude input is in km or m; need to check
def get_sunpy_el_az(ltd, lnd, alt, gps_time):
    c = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime=gps_time,
         observer="earth", frame=frames.Helioprojective)
    #Now we establish our location on the Earth
    gps_location = EarthLocation(lat=ltd*u.deg, lon=lnd*u.deg, height=alt*u.km)
    #Now lets convert this to a local measurement of Altitude and Azimuth.
    frame_altaz = AltAz(obstime=Time(gps_time), location=gps_location)
    sun_altaz = c.transform_to(frame_altaz)
    
    return [sun_altaz.T.alt.degree, sun_altaz.T.az.degree] #this is [el, az]

#this function converts a date in the original format (gps_timestamp = '130921_163855.049')
#to the form gps_timestamp = '2021-09-13 16:38:55.049'
def time_to_sunpy_format(gps_timestamp):
    year = gps_timestamp[4:6]
    month = gps_timestamp[:2]
    day = gps_timestamp[2:4]
    hour = gps_timestamp[7:9]
    minute = gps_timestamp[9:11]
    seconds_and_millis = gps_timestamp[11:]
    new_date = year + '-' + month + '-' + day + ' ' + hour + ':' + minute + ':' + seconds_and_millis
    #we are still returning the wrong thing at the end here, deliberately (it's just a random date)
    #since the test data that I was using doesn't have the dates in the right form to process
    #we should noramlly return the new_date variable, once the input date is in the expected form
    return new_date

#function form of Least Square fitting
#Y is dependant variable
#X is independant, unchanging, certain
# rmse is from variation in the Y variable
def least_square_fit(X,Y):
    mean_x = np.mean(X); mean_y = np.mean(Y)
    n = len(X)
    numer = 0; denom = 0
    for i in range (n):
        numer += (X[i] - mean_x) * (Y[i] - mean_y)
        denom += (X[i] - mean_x) ** 2
    m = numer / denom
    b = mean_y - (m * mean_x)
    rmse = 0
    for i in range(n):
        y_pred = b + m *X[i]
        rmse += (Y[i] - y_pred) ** 2
    rmse = np.sqrt(rmse/n)
    return [m, b, rmse]

#This function finds the cartesian equivalent of the pointing vectors of the cameras, 
#each with magnitude 1 
#useful for testing rotation functions to see if they are working properly, not used in actual GS
#input is in degrees;
# theta = inclination from the y-axis (not elevation, instead measure the angle from positive y to the pointing direction)
# phi = azimuth, such that z = 0 and x = 90 degrees
def sphere_to_cart(theta, phi):
    #convert to radians
    theta = theta/360*2*np.pi
    phi = phi/360*2*np.pi
    #compute cartesian coords
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions
    z = np.cos(phi)*np.sin(theta)
    x = np.sin(phi)*np.sin(theta)
    y = np.cos(theta)
    return [x,y,z]

#these functions take main magnetometer ADC values and convert them to microTesla
def main_mag_x_adc_to_uT(x, neg_x):
    diff_x = x - neg_x
    return 0.041 * diff_x + 0.0177
def main_mag_y_adc_to_uT(y, neg_y):
    diff_y = y - neg_y
    return 0.0411 * diff_y - 0.1047
def main_mag_z_adc_to_uT(z, neg_z):
    diff_z  = z - neg_z
    return 0.0413 * diff_z - 0.1516
    
#these functions take boom magnetometer ADC values and convert them to microTesla
def boom_mag_x_adc_to_uT(x, neg_x):
    diff_x = x - neg_x
    return 0.0411 * diff_x + 1.1819
def boom_mag_y_adc_to_uT(y, neg_y):
    diff_y = y - neg_y
    return 0.0419 * diff_y - 0.7078
def boom_mag_z_adc_to_uT(z, neg_z):
    diff_z  = z - neg_z
    return 0.0409 * diff_z + 0.2525

#this funciton takes temperature sensor ADC values and converts them to degrees Celsius
def temp_adc_to_deg(temp_adc):
    return temp_adc * 0.0633 - 76.801

#opening one of the files that we created from the HASP website, turning it into a usable df
def read_downlinked_csv(file_name):
    df = pd.read_csv(file_name, skiprows = 1, header = None, index_col = 0)
    #get rid of random weird useless rows:
    #does this by checking the length of the beginning of each row
    #the 000000, 000001, etc. is always 6 long, so anything else at the start of a row will be deleted
    for row in df.iterrows():
        if len(str(row[0])) != 6:
            try:
                df.drop(index = row[0], inplace = True)
            #keyerror excpetion because sometimes the row may have been deleted already
            #(and iterrows() is iterating through the original df, not the updated/dropped versions)
            except KeyError:
                pass
    return df

#this function takes latitude and longitude in the form ddmm.mmmm
#(first two are degree values, followed by degree minutes/decimal minutes)
#it converts these values to decimal degrees and determines the... 
# +/- sign based on EWNS direction
# W is negative for longitude and S is negative for Latitude; otherwise +
def dm_to_dd(lnd_or_ltd, direction):
    string = str(lnd_or_ltd)
    degrees = float(string[:2]) + float(string[2:])/60
    if direction == ('W' or 'S'):
        degrees = -degrees
    return degrees
    

