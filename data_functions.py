import numpy as np
import astropy.coordinates as coord
import astropy.units as u

def get_kinematics(ra, dec, distance, pm_ra, pm_dec, v_los):
    #_____coordinate transformation_____
    #input coordinates as quantities with units:
    # Get Gaia kinematics
    # ra       = np.array(Gaia_data['ra'])                *u.degree     #right ascension [deg] in ICRS
    # dec      = np.array(Gaia_data['dec'])               *u.degree     #declination [deg] in ICRS
    # # distance = 1./np.array(Gaia_data['parallax'])       *u.kpc        #distance from Sun [kpc] using parallax
    # distance = np.array(Gaia_data['distance_gspphot'])       *u.pc        #distance from Sun [pc] using gspphot distances


    # pm_ra    = np.array(Gaia_data['pmra'])              *u.mas/u.year #proper motion in direction of right ascension [mas/yr] in ICRS
    # pm_dec   = np.array(Gaia_data['pmdec'])             *u.mas/u.year #proper motion in direction of declination [mas/yr] in ICRS
    # v_los    = np.array(Gaia_data['radial_velocity'])   *u.km/u.s     #line-of-sight velocity [km/s]a
    position=[np.cos(25/180*np.pi)*8.2,np.sin(25/180*np.pi)*8.2]

    # Let's first input the 6D information to astrpy
    coordinates = coord.SkyCoord(
        ra=ra,
        dec=dec,
        distance=distance,
        pm_ra_cosdec=pm_ra,
        pm_dec=pm_dec,
        radial_velocity=v_los,
        frame='icrs' # This is defining the Reference Frame that our Coordinates are living in;
        # ICRS is the International Celestial Reference Frame, where the Galactic Centre is at
        # (RA,Dec) = (266.4051, -28.936175) deg
    )
    # In addition to the position of the Galactic Centre in (RA,Dec), we also have to give some information
    # on the position and motion of the Sun relative to it
    # astropy has its own default values, but we have actual measurements that we can use:

    r_sun_galactic_centre = 8.2*u.kpc # Gravity Collaboration, 2019, A&A, 625, 10
    phi_sun_galactic_centre = 0.0*u.rad # This is just for completeness
    z_sun_galactic_plane = 25.0*u.pc # Bland-Hawthorn & Gerhard, 2016, ARA&A, 54, 529
    # Reid & Brunthaler (2004, ApJ, 616, 872) have further measured the total angular motion of the Sun with respect to the Galactic Centre
    v_total_sun = (np.tan(6.379*u.mas)*r_sun_galactic_centre/u.yr).to(u.km/u.s) # pm_l by Reid & Brunthaler 2004, ApJ, 616, 872
    # And Schoenrich, Binney and Dehnen (2010, MNRAS, 403, 1829) have measured the motion of the Sun relative to the Local Standard of Rest

    v_peculiar = [11.1, 12.24, 7.25]*u.km/u.s # U and W from Schoenrich, Binney, Dehnen, 2010, MNRAS, 403, 1829, V so that V = V_total-V_sun
    # We can therefore find the Solar motion as
    galcen_v_sun = [11.1, v_total_sun.value, 7.25]*u.km/u.s
    print(galcen_v_sun)

    # Let's define the Galactocentric Reference Frame, but use our better value of the Distance
    galactocentric_frame= coord.Galactocentric(
        galcen_distance=r_sun_galactic_centre,
        galcen_v_sun=coord.CartesianDifferential(galcen_v_sun),
        z_sun=z_sun_galactic_plane
    )

    # Now let's transform the coordinates to the Galactocentric Cartesian Frame (X,Y,Z) relative to the Galactic Centre
    galcen_coordinates = coordinates.transform_to(galactocentric_frame)
    # If you want to look under the hood of how to go from RA,Dce,Distance to X,Y,Z via matrix multiplication, you can check out:
    # https://articles.adsabs.harvard.edu/pdf/1987AJ.....93..864J
    # If you want to work through the coordinate transformations yourself, I can also recommend Jo Bovy's coordinate transformation scripts
    # https://docs.galpy.org/en/v1.6.0/reference/bovycoords.html
    # and in particular https://docs.galpy.org/en/v1.6.0/reference/coordsradectolb.html to go from (RA,DEC) to (l,b) - which is by far the most difficult one

    # Now we also want to go from X,Y,Z to R,phi,z - which is more useful if we want to compare with other galaxies
    # 
    galcen_coordinates.representation_type = 'cylindrical'

    # Let's get an idea of how these coordinates look like
    # Note the slightly different notation:
    # R = rho
    # phi = phi
    # z = z

    # vR = d_rho
    # vphi (angular) = d_phi (but is an angular velocity noted in mas/yr, so has to be converted to km/s by multiplying with R)
    # vphi = d_phi * R
    # vz = d_z

    # print(galcen_coordinates)
    # Let's save the values in the correct units
    R_kpc   = galcen_coordinates.rho.to(u.kpc).value
    phi_rad = galcen_coordinates.phi.to(u.rad).value
    z_kpc   = galcen_coordinates.z.to(u.kpc).value

    vR_kms = galcen_coordinates.d_rho.to(u.km/u.s).value
    vT_kms = -(galcen_coordinates.d_phi.to(u.rad/u.s)*galcen_coordinates.rho.to(u.km)  / (1.*u.radian)).value
    vz_kms = galcen_coordinates.d_z.to(u.km/u.s).value
    L_Z=vT_kms*R_kpc

    galcen_coordinates.representation_type = 'cartesian'
    x_solar=galcen_coordinates.x
    y_solar=galcen_coordinates.y
    z_solar=galcen_coordinates.z
    #rotate axis to align the bar with x-axis
    theta=25/180*3.14 #sun at 25 deg 
    # the sun is located at (7.42,3.46)
    x_bar=-x_solar*np.cos(theta)-y_solar*np.sin(theta)
    y_bar=-x_solar*np.sin(theta)+y_solar*np.cos(theta)
    z_bar=z_solar

    x_bar=x_bar.to(u.kpc).value
    y_bar=y_bar.to(u.kpc).value
    z_bar=z_bar.to(u.kpc).value

    # Define the snd as a cylinder
    cdt_snd= ((x_bar-position[0])**2+(y_bar-position[1])**2<1**2) & (abs(z_bar)<1)

    return (x_bar,y_bar,z_bar),(vR_kms,vT_kms,vz_kms),(L_Z,vR_kms),cdt_snd