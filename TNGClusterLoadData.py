import illustris_python as il
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from datetime import datetime
import warnings

# Only see warnings once.
warnings.filterwarnings('once')

##### Global constants
# Dictionary to relate snap shot number and redshift.
snapRedshift = {2: 12, 3: 11, 4: 10, 6: 9, 8: 8, 11: 7, 13: 6, 17: 5,
                21: 4, 25: 3, 33: 2, 40: 1.5, 50: 1, 59: 0.7, 67: 0.5,
                72: 0.4, 78: 0.3, 84: 0.2, 91: 0.1, 99: 0}

##### Analysis-specific parameters.

# You can change these in order to produce data of interest for
# your project. For example, if you are interested in analyzing
# the stellar mass of the central galaxy within 200kpc of the 
# center, you can change `starBCGDistCut` to 200.
little_h = 0.6774 # Hubble constant H = 100little_h. UNITS: km/s/Mpc.
minM200c  = 10**(13) # Minimum halo mass in R200c. UNITS: M_sun
minSatMass = 1e10 # Minimum stellar mass for a subhalo to be 
                  # considered a satellite galaxy. UNITS: M_sun
starBCGDistCut = 100 # The distance threshold to consider stellar mass
                     # of BCG i.e. the central galaxy. UNITS: kpc
hotGasTemp = 1e6 # Temperature for gas to be considered hot. UNITS: Kelvin.
# initialize snap, redshift, and scale_factor.
snap = 99
redshift = snapRedshift[snap]
scale_factor = 1/(1 + redshift)
##### Physical Constants

K_B = 1.380649*10**(-16) # Boltzmann constant. UNITS: CGS (erg/K)
gamma = 5/3 # Thermodynamic adiabatic index
X_H = 0.76 # Hydrogen fraction
Mass_of_proton = 1.67262192369*10**-(24) # Mass of proton. UNITs: grams

##### Target simulation and snapshot/redshift from which to extract data.
# Simulation options: "TNG-Cluster", "TNG300-1", "TNG300-2", "TNG300-3".
simulation = "TNG-Cluster" # TNG simulation of interest
filePath = f"/virgotng/mpia/TNG-Cluster/L680n8192TNG/output" # sim file path 
                                                             # data files

##### Data Fields that will be used in the simulation.

# You can add to these fields but you just have to update
# the load data functions to include these changes.
haloFields = ['GroupMass', 'GroupFirstSub', 'GroupPos',
              'GroupVel', 'GroupMassType', 'Group_M_Crit200',
              'Group_M_Crit500','Group_R_Crit200', 'Group_R_Crit500',
              'GroupContaminationFracByMass', 'GroupContaminationFracByNumPart',
              'GroupOrigHaloID', 'GroupPrimaryZoomTarget']
subhaloFields = ['SubhaloMass', 'SubhaloGrNr', 'SubhaloMassType',
                 'SubhaloPos', 'SubhaloSpin', 'SubhaloVel']
gasFields = ['Coordinates', 'Density', 'ElectronAbundance', 
             'InternalEnergy', 'Masses',]
starFields = ['Coordinates', 'Masses']

##### Helper Functions 

# Helper function to find the distance between two points.
def distance(x, y, z, x0, y0, z0):
    """
    Take Euclidean coordinates of two points and find the distance between those
    two points.

    Parameters
    ----------
        x : float
            The x-coordinate of the first point. 

        y : float
            The y-coordinate of the first point. 

        z : float
            The z-coordinate of the first point. 

        x0 : float
            The x-coordinate of the second point. 

        y0 : float
            The y-coordinate of the second point. 

        z0 : float
            The z-coordinate of the second point. 


    Returns
    -------
        float
            The distance between two points.

    """

    dx = x - x0
    dy = y - y0
    dz = z - z0
    
    return np.sqrt(dx**2 + dy**2 + dz**2)


##### Data loading functions.

def loadHaloDF():
    """
    Load a DataFrame including halo data, specifically describing the quantities
    given in the global variable `haloFields`. You can change the global 
    variable `haloFields` and then change this function accordingly to include 
    the new quantity of interest.

    Parameters
    ----------
        None.
    
    Returns
    -------
        dfHalos: pandas.DataFrame
            DataFrame with columns describing the quantities
            in the global varaiable `haloFields` with a log mass scale.

    """

    # Load the halo data from the simulation
    halos = il.groupcat.loadHalos(basePath = filePath, snapNum = snap,
                                  fields = haloFields)
    
    # Create separate data arrays and change units.
    centGalaxyID = halos['GroupFirstSub']
    haloPos = halos['GroupPos'] * scale_factor/little_h # ckpc/h -> kpc
    haloVel = halos['GroupVel'] * 1/scale_factor # km/s/a -> km/s
    haloMassByType = halos['GroupMassType'] * 1e10/little_h  # 10^10M_sun/h 
                                                            #    -> M_sun
    haloMass = halos['GroupMass'] * 1e10/little_h
    haloM200 = halos['Group_M_Crit200'] * 1e10/little_h
    haloM500 = halos['Group_M_Crit500'] * 1e10/little_h
    haloR200 = halos['Group_R_Crit200'] * scale_factor/little_h #ckpc/h -> kpc
    haloR500 = halos['Group_R_Crit500'] * scale_factor/little_h

    contamByMass = halos['GroupContaminationFracByMass']
    contamByPart = halos['GroupContaminationFracByNumPart']
    origID = halos['GroupOrigHaloID']
    zoomTarget = halos['GroupPrimaryZoomTarget']

    # Split different dimensions of halo position and velocity.
    haloPosX, haloPosY, haloPosZ  = haloPos[:, 0], haloPos[:, 1], haloPos[:, 2]
    haloVelX, haloVelY, haloVelZ = haloVel[:, 0], haloVel[:, 1], haloVel[:, 2]
    # 0, 4, and 5 are the particle type numbers for gas, star, and BH
    # particles, respectively.
    haloGasMass = haloMassByType[:, 0]
    haloDMMass = haloMassByType[:, 1]
    haloStarMass = haloMassByType[:, 4]
    haloBHMass = haloMassByType[:, 5]

    # Create halo DataFrame using the above halo data extracted from the sim.
    haloKeys = ['contam_mass', 'contam_part', 'Original ID', 'Zoom target',
                'M_tot', 'M_200c', 'M_500c', 'R_200c', 'R_500c', 'M_gas',
                'M_dm', 'M_star', 'M_bh', 'pos_x', 'pos_y',
                'pos_z', 'vel_x', 'vel_y', 'vel_z', 'BCG ID']
    haloVals = [contamByMass, contamByPart, origID, zoomTarget,
                haloMass, haloM200, haloM500, haloR200, haloR500, haloGasMass,
                haloDMMass, haloStarMass, haloBHMass, haloPosX, haloPosY,
                haloPosZ, haloVelX, haloVelY, haloVelZ, centGalaxyID]
    dfHalos = pd.DataFrame(dict(zip(haloKeys, haloVals)))
    
    # Add a column to the halo DataFrame for the interger halo IDs which
    # corresponds to the index of the row that each halo occupies. This will be 
    # important when merging the halo and subhalo DataFrames.
    dfHalos.reset_index(inplace = True)
    dfHalos.rename(columns = {'index': 'Halo ID'}, inplace = True)

    # Use a log mass scale.
    dfHalos[['M_tot', 'M_500c', 'M_200c', 'M_gas','M_dm', 'M_star', 
             'M_bh']] = dfHalos[['M_tot', 'M_500c', 'M_200c', 'M_gas', 'M_dm', 
                                 'M_star', 'M_bh']].apply(np.log10, axis = 0)

    return dfHalos


# Load Subhalo DataFrame.
def loadSubDF():
    """
    Load a DataFrame including suhalo data, specifically describing the 
    quantities given in the global variable `subhaloFields`. You can change the 
    global variable `subhaloFields` and then change this function accordingly
    to include the new quantity of interest.

    Parameters
    ----------
        None.
    
    Returns
    -------
        dfSubs: pandas.DataFrame
            DataFrame with columns describing the quantities
            in the global varaiable `subhaloFields` with a log mass scale.

    """

    subhalos = il.groupcat.loadSubhalos(basePath = filePath, snapNum = snap, 
                                        fields = subhaloFields)
    
    # Create separate data arrays for subhalos and also change units.
    subParentID = subhalos['SubhaloGrNr']
    subMass = subhalos['SubhaloMass'] * 1e10/little_h
    subMassByType = subhalos['SubhaloMassType'] * 1e10/little_h # 10^10M_sun/h 
                                                                  # -> M_sun
    subPos = subhalos['SubhaloPos'] * scale_factor/little_h # ckpc/h -> kpc

    # Split different dimensions of subhalo position.
    subPosX, subPosY, subPosZ = subPos[:, 0], subPos[:, 1], subPos[:, 2]

    # 0, 4, and 5 are the particle type numbers for gas, star, and BH
    # particles, respectively.
    subGasMass = subMassByType[:, 0]
    subDMMass = subMassByType[:, 1]
    subStarMass = subMassByType[:, 4]
    subBHMass = subMassByType[:, 5]

    # Create subhalo DataFrame using the above subhalo data extracted from
    # the simulation.
    subKeys = ['M_tot', 'M_gas', 'M_dm', 'M_star', 'M_bh',
               'pos_x', 'pos_y', 'pos_z', 'Parent ID']
    subVals = [subMass, subGasMass, subDMMass, subStarMass, subBHMass,
               subPosX, subPosY, subPosZ, subParentID]

    dfSubs = pd.DataFrame(dict(zip(subKeys, subVals)))

    # Add a column to the subhalo DataFrame for the in integer subhalo IDs which
    # corresponds to the index of the row that each subhalo occupies.
    dfSubs.reset_index(inplace = True)
    dfSubs.rename(columns = {'index': 'Subhalo ID'}, inplace = True)

    # Use a log mass scale.
    dfSubs[['M_tot', 'M_gas', 'M_dm', 'M_star',
            'M_bh']] = dfSubs[['M_tot', 'M_gas','M_dm', 'M_star', 
                               'M_bh']].apply(np.log10, axis = 1)
    
    return dfSubs


# Load Gas data for either subhalo or halo.
def loadGasData(centPosition, haloID = -1, subhaloID = -1):
    """
    Load gas particle data from the simulation. We load the gas mass, 
    gas distance from halo, gas distance from subhalo if subhaloID is provided, 
    gas temperature and gas density. 
    You can change the  global variable `gasFields` and then change this
    function accordingly to include the new quantity of interest.

    Parameters
    ----------
        position: tuple or list
            Position of the halo/subhalo.

        haloID: signed int 
            Simulation ID number for the halo of interest.
            If the subhaloID is provided, then this is the parent halo 
            of the subhalo.

        subhaloID: signed int 
            Simulation ID number for the subhalo of interest. 

    
    Returns
    -------
        gasMass: numpy.ndarray  
            Array describing the mass of every gas particle.

        gasDistFromHalo: numpy.ndarray 
            Array describing the distance between the particle and the center
            of the halo that it belongs to for every gas particle.

        gasDistFromSub: numpy.ndarray 
            Array describing the distance between the particle and the center
            of the subhalo that it belongs to for every gas particle. Will be 
            None if subhaloID not provided.

        gasTemp: numpy.ndarray 
            Array describing the temperature of every gas particle.

        gasDensity : numpy.ndarray 
            Array describing the mass of every gas particle.

    """

    # Raise a TypeError when a subhalo ID is provided without its parent halo ID.
    if (subhaloID == -1 and haloID == -1):
        raise TypeError("Cannot have empty subhalo and halo IDs.")
    
    # Raise a TypeError when neither subhalo ID nor a halo ID is provided.
    if (subhaloID != -1 and haloID == -1):
        raise TypeError("Cannot input subhalo ID without this subhalo's" /
                        "parent halo ID.")
    
    # If subhalo ID is provided (by the above exceptions, this means that
    # the halo ID is also provided), then load gas data for subhalo, and find
    # the position the subhalo and halo.
    if subhaloID != -1:
        gases = il.snapshot.loadSubhalo(basePath = filePath, snapNum = snap, 
                                        id = subhaloID, partType = 'gas', 
                                        fields = gasFields)
        subPosX, subPosY, subPosZ = centPosition
        (haloPosX, haloPosY, 
         haloPosZ) = (il.groupcat.loadSingle(basePath = filePath,
                                            snapNum = snap,
                                            haloID = haloID)['GroupPos'] 
                                            * scale_factor/little_h)
    # Otherwise, only the haloID is provided and we load the gas data and obtain
    # the position of the halo.
    elif haloID != -1:
        gases = il.snapshot.loadHalo(basePath = filePath, snapNum = snap, 
                            id = haloID, partType = 'gas', fields = gasFields)
        haloPosX, haloPosY, haloPosZ = centPosition
    
    # We check if there are any gas particle, which is denoted by gases['count]
    # being nonzero.
    if gases['count'] > 0:
        # Create separate data arrays also change units.
        gasCoords = gases['Coordinates'] * scale_factor/little_h #ckpc/h -> kpc
        # Density units from (1e10 M_sun/h)/(ckpc/h)^3 to (M_sun/kpc^3)
        gasDensity = (gases['Density']/(scale_factor/little_h)**3
                      * (1e10/little_h))
        # ElectronAbundance and InternalEnergy will be used to
        # calculate the gas temperature.
        gasElectronAbundance = gases['ElectronAbundance']
        gasEnergy = gases['InternalEnergy'] # (km/s)^2
        gasMass = gases['Masses'] * 1e10/little_h # 10^10M_sun/h -> M_sun
        # Split different dimensions of gas position.
        gasPosX, gasPosY, gasPosZ = (gasCoords[:, 0], gasCoords[:, 1], 
                                     gasCoords[:, 2])
        # Calculate the gas Temperature in Kelvin.
        gasMeanMolecularWeight = (4/(1. + 3. * X_H + 
                                    4. * X_H * gasElectronAbundance)
                                    * Mass_of_proton)
        gasTemp = (gamma - 1) * gasEnergy/K_B * 10**10 * gasMeanMolecularWeight

        # Calculate distances for each gas particle from the center of the
        # halo/subhalo that it belongs to.
        gasDistFromSub = (distance(subPosX, subPosY, subPosZ, 
                                  gasPosX, gasPosY, gasPosZ) 
                                  if subhaloID != -1 else None)
        gasDistFromHalo = distance(haloPosX, haloPosY, haloPosZ,
                                   gasPosX, gasPosY, gasPosZ)
    # If there are not gas particles, then we set the masses to 0 and the 
    # distances to infinity.
    else:
        (gasMass, gasDistFromHalo, gasDistFromSub, 
             gasTemp, gasDensity) = (np.array([0]), np.array([float("inf")]),
                                     np.array([float("inf")]), 
                                     np.array([0]), np.array([0]))
    

    return gasMass, gasDistFromHalo, gasDistFromSub, gasTemp, gasDensity


def loadStarData(centPosition, haloID = -1, subhaloID = -1):
    """
    Load star particle data from the simulation. We load the star mass,
    star distance from halo, and star distance from subhalo 
    if subhaloID is provided.
    You can change the  global variable `starFields` and then change this
    function accordingly to include the new quantity of interest.

    Parameters
    ----------
        centPosition: tuple or list 
            Position of the center of the halo/subhalo.

        haloID: signed int
            Simulation ID number for the halo of interest. If the subhaloID
            is provided, then this is the parent halo of the subhalo.

        subhaloID : signed int
            Simulation ID number for the subhalo of interest. 

    
    Returns
    -------
        starMass : numpy.ndarray 
            Array describing the mass of every star particle.

        starDistFromHalo: numpy.ndarray 
            Array describing the distance between the particle and the center
            of the halo that it belongs to for every star particle.

        starDistFromSub : numpy.ndarray 
            Array describing the distance between the particle and the center 
            of the subhalo that it belongs to for every star particle. Will be 
            None if subhaloID not provided.

    """

    # Raise a TypeError when a subhalo ID is provided without its parent halo ID.
    if (subhaloID == -1 and haloID == -1):
        raise TypeError("Cannot have empty subhalo and halo IDs.")
    
    # Raise a TypeError when neither subhalo ID nor a halo ID is provided.
    if (subhaloID != -1 and haloID == -1):
        raise TypeError("Cannot input subhalo ID without this subhalo's" /
                        "parent halo ID.")
    
    # If subhalo ID is provided (by the above exceptions, this means that
    # the halo ID is also provided), then load star data for subhalo, and find
    # the position the subhalo and halo.

    if subhaloID != -1:
        stars = il.snapshot.loadSubhalo(basePath = filePath, snapNum = snap, 
                                        id = subhaloID, partType = 'star', 
                                        fields = starFields)
        subPosX, subPosY, subPosZ = centPosition
        (haloPosX, haloPosY, 
         haloPosZ) = (il.groupcat.loadSingle(basePath = filePath,
                                            snapNum = snap,
                                            haloID = haloID)['GroupPos'] 
                                            * scale_factor/little_h)
    # Otherwise, only the haloID is provided and we load the gas data and obtain
    # the position of the halo.
    elif haloID != -1:
        stars = il.snapshot.loadHalo(basePath = filePath, snapNum = snap, 
                            id = haloID, partType = 'star', fields = starFields)
        haloPosX, haloPosY, haloPosZ = centPosition

    # We check if there are any star particle, which is denoted by stars['count]
    # being nonzero.
    if stars['count'] > 0:
        # Create separate data arrays also change units.
        starCoords = stars['Coordinates'] * scale_factor/little_h # ckpc/h ->kpc
        starMass = stars['Masses'] * 1e10/little_h # 10^10M_sun/h -> M_sun
        # Split different dimensions of star position.
        starPosX, starPosY, starPosZ = (starCoords[:, 0], starCoords[:, 1], 
                                        starCoords[:, 2])
        # Calculate distances for each star particle from the center of the
        # halo/subhalo that it belongs to.
        starDistFromSub = (distance(subPosX, subPosY, subPosZ, 
                                  starPosX, starPosY, starPosZ) 
                                  if subhaloID != -1 else None)
        starDistFromHalo = distance(haloPosX, haloPosY, haloPosZ,
                                   starPosX, starPosY, starPosZ)
    # If there are not star particles, then we set the masses to 0 and the 
    # distances to infinity.
    else:
        (starMass, starDistFromHalo, 
         starDistFromSub) = (np.array([0]), np.array([float("inf")]), 
                             np.array([float("inf")]))
    
    return starMass, starDistFromHalo, starDistFromSub


##### Loading the DataFrames for Halos and Subhalos.

# We will load two DataFrames:
# 1) A halo DataFrame that includes both halo data and particle (gas and star)
#    data for these halos merged together.
# 2) A subhalo DataFrame that includes both halo data and particle (gas
#    and star) data for these halos merged together.
if __name__ == "__main__":
    snap = int(sys.argv[1])
    redshift = snapRedshift[snap]
    scale_factor = 1/(1 + redshift)
    # Start timer.
    start = datetime.now()

    print("\nLoading halo DataFrame:\n")
    # Load halo DataFrame.
    dfHalos = loadHaloDF()
    # Filter halos by halo M200c greater than minimum listed in 
    # constants section.
    dfHalos = dfHalos[dfHalos['M_200c'] > np.log10(minM200c)]
    # Reset the index to be consecutive numbers after filtering.
    dfHalos.set_index(np.arange(len(dfHalos)), inplace = True)

    # Add particle data to halo data.
    for row, haloID in enumerate(tqdm(dfHalos['Halo ID'])):
        position = (dfHalos.loc[row, 'pos_x'], dfHalos.loc[row, 'pos_y'], 
                    dfHalos.loc[row, 'pos_z'])
        gasMass, haloGasDistFromHalo, _, gasTemp, _ = loadGasData(position, 
                                                    haloID = haloID)
        starMass, haloStarDistFromHalo, _ = loadStarData(position, 
                                                         haloID = haloID)

        R200 = dfHalos.loc[row, 'R_200c']
        gasIn200 = np.where(haloGasDistFromHalo < R200)
        starIn200 = np.where(haloStarDistFromHalo < R200)
        hotGasIn200 = np.where((haloGasDistFromHalo < R200) 
                               & (gasTemp > hotGasTemp))

        dfHalos.loc[row, 'M_star_200c'] = np.log10(np.sum(starMass[starIn200]))
        dfHalos.loc[row, 'M_gas_200c'] = np.log10(np.sum(gasMass[gasIn200]))
        dfHalos.loc[row, 'M_gas_200c_hot'] = \
            np.log10(np.sum(gasMass[hotGasIn200]))
        # If there is no gas in the halo (i.e. gas mass is zero), then 
        # the weights will all some to zero and we will get a ZeroDivisionError.
        try:
            dfHalos.loc[row, 'avg_temp_M200c'] = \
                np.log10(np.average(gasTemp[gasIn200], 
                                    weights = gasMass[gasIn200]))
        except ZeroDivisionError:
            dfHalos.loc[row, 'avg_temp_M200c'] = float("-inf")
        
    print("\nLoading subhalo DataFrame:\n")
    # Load subhalo DataFrame.
    dfSubs = loadSubDF()
    # Merge halo DataFrame and subhalo DataFrame based on the the halo ID and
    # subhalo Parent ID (describing what halo this subhalo belongs to).
    dfSubs = dfSubs.merge(dfHalos[['Halo ID', 'R_200c', 'BCG ID']],
                          how = 'inner', left_on = 'Parent ID', 
                          right_on = 'Halo ID')
    # Create a new column in subhal DataFrame that describes if this
    # subhalo is the primary subhalo of the parent halo.
    dfSubs['is_primary'] = (dfSubs['Subhalo ID'] == dfSubs['BCG ID'])
    # Drop Halo ID column because it's use was just to merge and we only
    # include subhalo data. We also drop the ID of the primary subhalo `BCG ID`
    # because that information is now included in the column `is_primary`.
    dfSubs.drop(labels = ['Halo ID', 'BCG ID'], axis = 1, inplace = True)
    dfSubs.set_index(np.arange(len(dfSubs)), inplace = True)

    for row, (haloID, 
              R200c, subID) in tqdm(enumerate((zip(dfSubs['Parent ID'], 
                                                   dfSubs['R_200c'], 
                                                   dfSubs['Subhalo ID']))),
                                                   total = len(dfSubs)):
        position = (dfSubs.loc[row, 'pos_x'], dfSubs.loc[row, 'pos_y'], 
                    dfSubs.loc[row, 'pos_z'])
        (subStarMass, subStarDistFromHalo, 
            subStarDistFromSub) = loadStarData(position, subhaloID = subID, 
                                               haloID = haloID)
        
        starInBCGCut = np.where(subStarDistFromSub < starBCGDistCut)
        starInR200c = np.where(subStarDistFromHalo < R200c)

        dfSubs.loc[row, 'M_star_tot'] = np.log10(np.sum(subStarMass))
        dfSubs.loc[row, f'M_star_{starBCGDistCut}kpc'] = \
            np.log10(np.sum(subStarMass[starInBCGCut]))
        dfSubs.loc[row, 'M_star_in_R200c'] = \
            np.log10(np.sum(subStarMass[starInR200c]))

    # Save the resultant DataFrames.
    dfHalos.to_csv(f'{simulation}-halo-catalog-snap{snap:02d}.csv',
                   index = False)
    dfSubs.to_csv(f'{simulation}-subhalo-catalog-snap{snap:02d}.csv',
                  index = False)

    # Calculate and display time in hrs:mins:secs
    end = datetime.now()
    totalSecs = (end - start).seconds
    hrs= totalSecs//3600
    mins = (totalSecs//60) % 60
    secs = totalSecs % 60
    print(f"\nDone! Time elapsed: {hrs:02d}:{mins:02d}:{secs:02d}\n")