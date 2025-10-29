import shutil
import numpy as np 
from perlin_noise import PerlinNoise
import random
from netCDF4 import Dataset
import subprocess
import time
import os
import pandas as pd
import xarray as xr
from tqdm import tqdm
import warnings
from scipy.stats import weibull_max
warnings.filterwarnings("ignore", category=DeprecationWarning)
from meshkernel import py_structures
from graph_creation import center_grid_graph, interpolate_variable, Mesh
from graph_creation import create_mesh_dhydro, generate_random_polygon_with_dike, save_mesh

def create_raw_dataset_folder(folder_name):
    """Create a folder for the raw datasets which contains 
    "DEM", "Geometry", "Hydrograph", and "Simulations" folders.
    
    If the folder exists, raises an error."""

    if os.path.exists(folder_name):
        print("Folder already exists")
    else:
        os.makedirs(folder_name)
        os.makedirs(os.path.join(folder_name, "DEM"))
        os.makedirs(os.path.join(folder_name, "Geometry"))
        os.makedirs(os.path.join(folder_name, "Hydrograph"))
        os.makedirs(os.path.join(folder_name, "Simulations"))

    return None

def mesh_DEM_generator(seed, mesh, grid_size=100,
                       noise_octave=(1,8), DEM_multiplier=(1,7), 
                       slope_multiplier=(0,0.01)):
    '''
    Generates a random digital elevation model (DEM) based on Perlin noise
    ------
    seed: int, replicable randomness for Perlin noise and magnitude randomizer
    mesh: meshkernel.py_structures.Mesh2d or Mesh object (contains mesh nodes and faces coordinates)
    grid_size: float, factor used to reduce the noise computations (higher values reduce the frequency of points)
    noise_octave: float, Perlin noise octave magnitude
    initial_octave: float, minimum Perlin noise octave magnitude
    DEM_multiplier: float, multiplier for max and min DEM values
    initial_DEM: float, minimum DEM value
    slope_multiplier: float, multiplier for slope
    dike: list, list of nodes that form the dike
    DEM_dike: float, if not None, assigns elevation to dike faces
    '''
    random.seed(seed)

    octaves = noise_octave[1]*random.random()+noise_octave[0]
    noise = PerlinNoise(octaves=octaves, seed=seed)
    DEM_multiplier = DEM_multiplier[1]*random.random()+DEM_multiplier[0]
    slope_multiplier = slope_multiplier[1]*random.random()+slope_multiplier[0]
    slope_direction = random.random()*360

    extent_min = int(min(mesh.node_x.min(), mesh.node_y.min())/grid_size)
    extent_max = int(max(mesh.node_x.max(), mesh.node_y.max())/grid_size)
    diff = extent_max - extent_min
    
    DEM = np.array([[noise([i/diff, j/diff]) for j in range(extent_min, diff)] for i in range(extent_min, diff)])
    slope = np.array([[i*np.cos(slope_direction) + j*np.sin(slope_direction) for j in range(extent_min, diff)] for i in range(extent_min, diff)])

    DEM = DEM*DEM_multiplier + slope*slope_multiplier

    num_grids = diff - extent_min
    _, grid_nodes = center_grid_graph(num_grids, num_grids, grid_size)
    DEM = interpolate_variable(mesh.face_xy, grid_nodes, DEM.reshape(-1), method='nearest')

    # # Add dike
    # if dike is not None:
    #     dike_nodes = np.concatenate([np.where((mesh.mesh_nodes == i).sum(1) == 2) for i in dike]).squeeze()
    #     dike_faces = np.where([sum([node in dike_nodes for node in face]) > 2 for face in mesh.face_nodes])[0]

    #     if DEM_dike is not None:
    #         random.seed(seed+1)
    #         DEM_dike = DEM_multiplier*random.random()+initial_DEM

    #     DEM[dike_faces] = DEM_dike

    return DEM

def change_nc_file(grid_file, varname, new_value):
    '''
    Change NETCDF variable 'varname' with 'new_value'
    '''
    ncfile = Dataset(grid_file,mode='r+') 

    ncfile.variables[varname][:] = new_value

    ncfile.close()
    return None

def save_DEM(DEM, dst_file, node_coords=None):
    '''
    Saves DEM file as .xyz or .txt file for numerical simulations
    ------
    DEM: np.array, elevetion map in x and y directions
    dst_file: str (path-like), destination file for saving DEM
    pos: dict, node positions (x, y)
    '''
    if node_coords is None:
        number_grids = DEM.shape[0] #int(len(pos)**0.5)
        y_grid = np.array([[(i+0.5) for j in range(number_grids)] for i in range(number_grids)])
        x_grid = y_grid.T
        xyz = np.array([[x, y, z] for x, y, z in zip(x_grid.reshape(-1,1), y_grid.reshape(-1,1), DEM.reshape(-1,1))]).squeeze()
    else:
        xyz = np.array([[x, y, z] for (x, y), z in zip(node_coords, DEM)])
    
    #creating xyz file for DEM with proper number of decimals
    np.savetxt(dst_file, xyz, fmt = ('%1.1f', '%1.1f', '%1.5f'))

    return None

def change_breach_location_mesh(breach_polygon_file, coords):
    """Changes the boundary condition polygon file according to the new breach location
    -------
    breach_polygon_file: str (path-like), location of the DHYDRO boundary condition polygon file
    """
    replacement = 'HydrographQ\n'\
              '    2    2\n'\
              f'{coords[0,0]}    {coords[0,1]}    HydrographQ_0001\n'\
              f'{coords[1,0]}    {coords[1,1]}    HydrographQ_0002'

    with open(breach_polygon_file, "w") as f:
        f.write(replacement)
        
    return None

def select_random_boundary_location(mesh, seed):
    """Selects a random edge on the boundary of the domain for the breach boundary condition
    -------
    mesh: meshkernel.py_structures.Mesh2d object
    seed: int, seed for random selection of boundary edge

    Returns: x and y coordinates of the breach location (np.array)
    """    
    np.random.seed(seed)

    boundary_edges = np.where((mesh.edge_faces.reshape(-1,2) == -1).sum(1) == 1)[0]
    breach_edge_id = np.random.random_integers(len(boundary_edges)-1)
    boundary_edge = mesh.edge_nodes.reshape(-1,2)[boundary_edges[breach_edge_id]]

    coords = mesh.mesh_nodes[boundary_edge]

    return coords

def generate_hydrograph(shape:float, peak_value:float, 
                        min_discharge:float, total_time:float, 
                        time_resolution:float, shape_max=3, shape_min=0):
    """Generates a hydrograph based on a Weibull distribution:
    shape: float, shape parameter of the Weibull distribution
        high values correspond to right-skewed hydrographs
        small values correspond to left-skewed hydrographs
    peak_value: float, peak value of the hydrograph
    min_discharge: float, minimum discharge value
    total_time: float, total time of the hydrograph [seconds]
    time_resolution: float, temporal resolution of the hydrograph [seconds]
    shape_max: float, maximum shape parameter
    shape_min: float, minimum shape parameter
    
    Example:
    x, y = generate_hydrograph(5, 100, 0, 48*3600, 3600)
    plt.plot(y)
    """
    shape = (shape < shape_min)*shape_min + \
            (shape > shape_max)*shape_max + \
            (shape <= shape_max and shape >= shape_min)*shape

    shape = 5+10**shape
    time_steps = int(total_time/time_resolution)+1
    time_x = np.linspace(weibull_max.ppf(0.01, shape), weibull_max.ppf(0.999, shape), time_steps)
    y = weibull_max.pdf(time_x, shape)

    time_x -= time_x.min()
    time_x = time_x/time_x.max() * total_time
    y = y/y.max() * (peak_value-min_discharge) + min_discharge
    
    return time_x, y

def generate_random_hydrograph(seed, total_time, time_resolution, min_discharge=0,
                               peak=(100,250), shape=(0,3)):
    """Generates a random hydrograph based on a Weibull distribution where 
    the shape parameter and the peak are randomly selected"""
    random.seed(seed)
    
    shape = shape[1]*random.random() + shape[0]
    peak_value = (peak[1]-peak[0])*random.random() + peak[0]

    return generate_hydrograph(shape, peak_value, min_discharge, total_time, time_resolution)

def save_hydrograph(hydrograph_time_series, dst_file):
    '''Saves hydrograph file as time series'''
    np.savetxt(dst_file, hydrograph_time_series, fmt = ('%1d', '%4.4f'))
    return None

def add_boundary_condition(boundary_condition_file, time_series, init_time, BC_type=2):
    """Assigns breach boundary condition
    BC_type:
    1: water level
    2: discharge
    """
    BC_dict = {1: "waterlevelbnd", 2: "dischargebnd"}
    unit_dict = {1: "m", 2: "m3/s"}

    replacement = f'[forcing]\n\
    Name				= HydrographQ_0001\n\
    Function                        = timeseries\n\
    Time-interpolation              = linear\n\
    Quantity                        = time\n\
    Unit                            = seconds since {init_time}\n\
    Quantity                        = {BC_dict[BC_type]}\n\
    Unit                            = {unit_dict[BC_type]}\n'

    for time, y in time_series:
        replacement += f'{str(time)}\t {str(y)}\n'
        
    with open(boundary_condition_file, "w") as f:
        f.write(replacement)

    return None

def run_simulation(model_folder):
    '''
    Run D-Hydro simulation, give model folder location
    Returns computational time
    '''
    #paths
    input_folder = os.path.abspath(os.path.join(model_folder, os.pardir))
    execution_file = f'{input_folder}\\run.bat'

    start_time = time.time()

    #Run D-Hydro, let Python wait till D-Hydro is done
    command = subprocess.Popen(execution_file, cwd = input_folder)
    command.wait()

    computation_time = round(time.time() - start_time, 4)

    return computation_time


def from_output_nc_to_txt(output_map, save_folder, seed):
    '''
    Converts numerical simulation in .nc file (output_map) to water depth and velocities .txt files
    ------
    output_map: str (path-like), netcdf output file location
    save_folder: str (path-like), folder location for saving results
    seed: int, simulation number
    '''
    #retrieve map data
    nc_dataset = xr.open_dataset(output_map)
    mesh2d_face_x = nc_dataset['Mesh2d_face_x'].data
    mesh2d_face_y = nc_dataset['Mesh2d_face_y'].data

    df = pd.DataFrame({'xloc': mesh2d_face_x, 'yloc': mesh2d_face_y})
    order = df.sort_values(['xloc', 'yloc']).index

    #extract water depth and velocities
    waterdepth = nc_dataset['Mesh2d_waterdepth'].data
    velocity_x = nc_dataset['Mesh2d_ucx'].data
    velocity_y = nc_dataset['Mesh2d_ucy'].data
    
    #saving water depth and velocities
    np.savetxt(f'{save_folder}/WD/WD_{seed}.txt', waterdepth, fmt='%1.4f')
    np.savetxt(f'{save_folder}/VX/VX_{seed}.txt', velocity_x, fmt='%1.4f')
    np.savetxt(f'{save_folder}/VY/VY_{seed}.txt', velocity_y, fmt='%1.4f')

    return None

def get_mdu_as_dict(config_file): 
    """Reads a .mdu file and returns it as a dictionary.
    This file contains the configuration of the model."""
    with open(config_file) as f:
        config = {}
        for line in f:
            if '=' in line:
                key, value = line.split('=')
                config[key.strip()] = value.strip()
    return config
    
def run_simulations_mesh(n_sim, model_folder, save_folder, start_sim=1, 
                         DEM_file=None, polygon_file=None, breach_coords=None,
                         noise_octave=(3,8), DEM_multiplier=(1,7), slope_multiplier=(0,0.001),
                         num_vertices_polygon=(20,30), number_of_multiscales=4, ellipticality=(1,2), grid_size=100,
                         random_hydrograph=False, hydrograph=None, min_discharge=0, peak=(50,150), shape=(0,2)):
    '''
    Run multiple hydraulic simulations for a mesh
    ------
    n_sim: int, number of simulations to run
    model_folder: str, directory containing the dimr_config file
    save_folder: str, directory in which to store simulations
    start_sim: int, starting simulation id (for replicability)
    polygon_file: str, name of the polygon file (default=None, random polygon is generated)
    DEM_file: str, name of the DEM file (default=None, random DEM is generated)
    breach_coords: np.array, breach coordinates for the boundary condition
    Mesh parameters:
        num_vertices_polygon: tuple of ints, minimum and maximum number of vertices for the polygon
        grid_size: float, multiplier for length of each cell (for None polygons)
        number_of_multiscales: int, number of mesh scales and refinement iterations
        ellipticality: tuple of floats, minimum and maximum ellipticality of the mesh (1 = circle)
    DEM parameters:
        noise_octave: tuple of floats, minimum and maximum intensity in spatial variation of DEM bumps
        DEM_multiplier: tuple of floats, minimum and maximum multiplier for DEM
        slope_multiplier: tuple of floats, minimum and maximum multiplier for slope
    hydrograph parameters:
        random_hydrograph: bool, if True, generates random hydrographs
        hydrograph: np.array, hydrograph time series (default=None)
        min_discharge: float, minimum discharge value (default = 0)
        peak: tuple of floats, minimum and maximum peak discharge
        shape: tuple of floats, minimum and maximum shape parameter (higher values correspond to left-skewed hydrographs)
    '''
    output_map = f'{model_folder}\\output\\FlowFM_map.nc'
    mesh_file = f'{model_folder}\\SWE_GNN_mesh_net.nc'
    breach_polygon_file = f'{model_folder}\\HydrographQ.pli'
    boundary_condition_file = f'{model_folder}\\Discharge.bc'
    boundary_polygon_file = polygon_file
    
    config_file = f'{model_folder}\\FlowFM.mdu'
    config = get_mdu_as_dict(config_file) 

    init_time = pd.to_datetime(config['RefDate'])
    simulated_time_seconds = int(config['TStop'])
    simulated_time_hours = simulated_time_seconds/3600
    temporal_resolution = int(config['MapInterval'])

    simulation_stats = []

    i = 0 # flag for mesh creation
    for sim in tqdm(range(start_sim, start_sim+n_sim)):
        sim = sim + 999 if sim == 91 else sim # change simulation 26 since it doesn't work
        if polygon_file is None:
            # Generate random polygon for mesh creation
            generate_random_polygon_with_dike(save_polygon=True, avg_radius=100*grid_size, irregularity=0.2, 
                                            spikiness=0.08, seed=sim, num_vertices=num_vertices_polygon, ellipticality=ellipticality)
            boundary_polygon_file = 'random_polygon.pol'

        # Create mesh
        if i==0:
            # this is to avoid creating the mesh multiple times for the same DEM
            mesh = create_mesh_dhydro(boundary_polygon_file, number_of_multiscales)
            save_mesh(mesh, mesh_file)
            if DEM_file is not None:
                i += 1

        # Generate DEM 
        if DEM_file is None:
            mesh_DEM = mesh_DEM_generator(sim, mesh=mesh, 
            noise_octave=noise_octave, DEM_multiplier=DEM_multiplier, 
            slope_multiplier=slope_multiplier)

            # Save DEM and mesh
            save_DEM(mesh_DEM, f'{model_folder}\\DEM.xyz', mesh.face_xy)
        else:
            DEM = np.loadtxt(DEM_file)[:,2].reshape(-1)
            grid_nodes = np.loadtxt(DEM_file)[:,:2]
            mesh_DEM = interpolate_variable(mesh.face_xy, grid_nodes, DEM, method='linear')

        # Change boundary condition location and hydrograph
        if breach_coords is None:
            breach_coords = select_random_boundary_location(mesh, seed=sim)
        change_breach_location_mesh(breach_polygon_file, breach_coords)
        breach_coords = None

        # Generate hydrograph
        if random_hydrograph:
            time_x, y = generate_random_hydrograph(sim, simulated_time_seconds, temporal_resolution, 
                                                    min_discharge=min_discharge, peak=peak, shape=shape)
        elif hydrograph is not None:
            time_x, y = hydrograph
        else:
            time_x, y = generate_hydrograph(shape[1], peak[1], min_discharge, simulated_time_seconds, temporal_resolution)
            y = y/y * peak[1]
        
        # Add hydrograph
        time_series = np.stack((time_x.round(0), y.round(4)), 1)
        add_boundary_condition(boundary_condition_file, time_series, init_time, BC_type=2)

        # Save DEM and hydrograph
        sim = sim - 999 if sim == 999+91 else sim # change back simulation id 26
        save_DEM(mesh_DEM, f'{save_folder}\\DEM\\DEM_{sim}.xyz', mesh.face_xy)
        save_hydrograph(time_series, f'{save_folder}\\Hydrograph\\Hydrograph_{sim}.txt')

        # run simulation
        computation_time = run_simulation(model_folder)

        # save results in files and overview folder
        shutil.copy(output_map, f'{save_folder}\\Simulations\\output_{sim}_map.nc')
        shutil.copy(boundary_polygon_file, f'{save_folder}\\Geometry\\polygon_{sim}.pol')
        simulation_stats.append([sim, mesh.face_x.shape[0], simulated_time_hours, computation_time])

    df = pd.DataFrame(simulation_stats, columns=['seed', 'mesh_num_faces', 'simulation_time[h]', 'computation_time[s]'])
    df.to_csv(f'{save_folder}\\overview.csv', mode='a', sep = ',', index = False, header=not os.path.exists(f'{save_folder}\\overview.csv'))

    return df