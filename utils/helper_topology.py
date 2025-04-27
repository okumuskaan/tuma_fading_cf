"""
Helper Functions for Topology Setup and Visualization

This file provides functions for setting up and visualizing system topologies in the TUMA framework.
It supports rectangular grids, hexagonal grids, and user placements with optional jitter
and multi-zone configurations.

Functions:
- hexagon_corners: Calculates corners of a hexagon given its side length and center.
- generate_hexagon_grid_corners: Generates corners for all hexagons in a grid.
- uniform_partition_hexagon: Divides a large hexagon into smaller hexagons.
- generate_random_positions_in_hexagons: Generates random user positions inside hexagons.
- plot_hexagon_topology: Visualizes a hexagonal grid with user positions and grid points.
- plot_topology: Plots general topologies, including grid-based and hexagonal setups.
- setup_topology: Configures the system topology based on selected parameters.

Usage:
This module is intended for use in the TUMA framework and can be imported as:
    from utils.helper_topology import setup_topology, plot_topology

Author: Kaan Okumus
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt

def hexagon_corners(side, center):
    """
    Calculate the corners of a single hexagon.

    Parameters:
    side : float
        Length of each side of the hexagon.
    center : complex
        Center of the hexagon represented as a complex number (real part is x, imag part is y).

    Returns:
    np.ndarray
        Array of complex numbers representing the corners of the hexagon.
    """
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] 
    return center + side * (np.cos(angles) + 1j * np.sin(angles))

def generate_hexagon_grid_corners(side, rows, cols):
    """
    Generate the corners of all hexagons in a grid.

    Parameters:
    side : float
        Length of each side of the hexagon.
    rows : int
        Number of rows.
    cols : int
        Number of columns.

    Returns:
    np.ndarray
        Array of hexagon corner coordinates.
    """
    dx = 3 * side / 2
    dy = np.sqrt(3) * side
    all_corners = []

    for row in range(rows):
        for col in range(cols):
            x_offset = col * dx
            y_offset = row * dy + (dy / 2 if col % 2 else 0)
            center = x_offset + 1j * y_offset
            corners = hexagon_corners(side, center)
            all_corners.append(corners)

    return np.array(all_corners)

def uniform_partition_hexagon(corners, n):
    """
    Uniformly partition a large hexagon into smaller hexagons.

    Parameters:
    corners : np.ndarray
        Corners of the large hexagon.
    n : int
        Number of smaller hexagons along one edge.

    Returns:
    np.ndarray
        Centers of smaller hexagons.
    """
    center = np.mean(corners)
    side = np.abs(corners[0] - corners[1])
    small_side = side / n
    circumradius = side

    dx = 3 * small_side / 2
    dy = np.sqrt(3) * small_side
    small_hex_centers = []

    for row in range(-n + 1, n):
        for col in range(-n + 1, n):
            x_offset = col * dx
            y_offset = row * dy + (dx / 2 if col % 2 else 0)
            candidate_center = center + x_offset + 1j * y_offset
            if np.abs(candidate_center - center) <= circumradius - small_side:
                small_hex_centers.append(candidate_center)

    return np.array(small_hex_centers)

def generate_random_positions_in_hexagons(hexagon_corners_array, num_positions=20):
    """
    Generate random positions inside each hexagon.

    Parameters:
    hexagon_corners_array : np.ndarray
        Corners of all hexagons.
    num_positions : int
        Number of random positions per hexagon.

    Returns:
    list
        List of arrays of random points for each hexagon.
    """
    random_positions = []

    for corners in hexagon_corners_array:
        corners = np.array(corners)
        center = np.mean(corners)
        triangles = [(center, corners[i], corners[(i + 1) % 6]) for i in range(6)]

        positions = []
        for _ in range(num_positions):
            triangle = triangles[np.random.randint(0, 6)]
            r1, r2 = np.random.rand(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            p = (1 - r1 - r2) * triangle[0] + r1 * triangle[1] + r2 * triangle[2]
            positions.append(p)

        random_positions.append(np.array(positions))

    return random_positions

def plot_hexagon_topology(hexagon_corners_array, user_positions, small_hexagon_positions, hex_color="orange", user_marker="x", grid_color="black"):
    """
    Visualize a hexagonal topology with user and zone positions.

    Parameters:
    hexagon_corners_array : np.ndarray
        Array of hexagon corners.
    user_positions : list
        List of user positions.
    small_hexagon_positions : list
        List of zone centers.
    hex_color : str
        Color of hexagon corners.
    user_marker : str
        Marker for users.
    grid_color : str
        Color of hexagon edges.
    """
    plt.figure(figsize=(8, 8))

    for hex_corners, positions, hex_grids in zip(hexagon_corners_array, user_positions, small_hexagon_positions):
        x = np.append(hex_corners.real, hex_corners.real[0])
        y = np.append(hex_corners.imag, hex_corners.imag[0])
        plt.plot(x, y, grid_color, linewidth=0.5)
        plt.scatter(hex_corners.real, hex_corners.imag, color=hex_color, s=50)
        plt.scatter(positions.real, positions.imag, s=15, marker=user_marker)

    plt.gca().set_aspect('equal')
    plt.title("Hexagonal Topology")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.axis("off")
    plt.show()


def setup_topology(side, topology_type=2, jitter=0, multiple_zone=True, force_U=None, force_B=None, mult=1, rows=4, cols=5):
    """
    Configure the system topology for TUMA framework.

    Parameters:
    side : float
        Side length for zones or cells.
    topology_type : int
        Type of topology (0: single-zone, 1: grid, 2: rectangular, 3: hexagonal).
    jitter : float
        Random displacement added to user positions.
    multiple_zone : bool
        Whether to create multiple zones.
    force_U : int
        Force a specific number of zones.
    force_B : int
        Force a specific number of users per zone.
    mult : int
        Sub-grid factor for type-2.
    rows : int
        Number of rows (for type-2 and type-3).
    cols : int
        Number of columns (for type-2 and type-3).

    Returns:
    (U, B, zone_centers, nus)
        U: number of zones
        B: number of users
        zone_centers: zone center coordinates
        nus: user coordinates
    """
    if multiple_zone==False:
        zone_centers = np.array([0.0+1j*0.0])
    if topology_type==0:
        if multiple_zone:
            raise ValueError(f"In topology type = {topology_type}, you cannot make multiple zone! It's only valid for single zone.")
        else:
            nus = np.array([0+1j*0])

    elif topology_type==1:
        # Setup a linear or 2D grid of zones
        if multiple_zone:
            if force_U is None or force_U==3:
                zone_centers = np.arange(-side, side*(1.0000001), side) + 1j*0.0
            else:                
                if force_U==9:
                    zone_centers_x = np.arange(-1,2)
                    zone_centers = np.sum([ar.astype(float) * (1-i + 1j*i) for i, ar in enumerate(np.meshgrid(zone_centers_x, zone_centers_x))],axis=0)*side
                    zone_centers = zone_centers.reshape(-1)
                else:
                    raise NotImplementedError(f"For topology type = 1, force_U can only take either 3 or 9, but you entered force_U = {force_U}.")
        if force_B is not None:
            B = force_B
        else:
            B = 16
        U = len(zone_centers)
        nus = np.zeros(B*U, dtype=complex)
        for u, zone_center in enumerate(zone_centers):
            nus[u*B:(u+1)*B] = zone_center + (side/np.sqrt(2)) * np.exp(1j * (np.linspace(0, 2 * np.pi, B, endpoint=False) + np.pi/4))

    elif topology_type==2:
        # Setup rectangular grid topology
        if multiple_zone==False:
            rows=1
            cols=1
        zone_centers_x = np.arange(cols)-(cols+1)/2 + 1
        zone_centers_y = np.arange(rows)-(rows+1)/2 + 1
        zone_centers = (np.sum([ar.astype(float) * (1-i + 1j*i) for i, ar in enumerate(np.meshgrid(zone_centers_x, zone_centers_y))],axis=0)*side).flatten()
        # Generate sub-grid of users
        nus_all = []
        nusbig = np.arange(cols+1,step=1)*side
        nussmall = (np.arange(rows+1/mult-1/(mult*10),step=1/mult)*side).reshape(-1,1)
        nusbig = nusbig - nusbig.mean()
        nussmall = nussmall - nussmall.mean()
        nus_all.append((nusbig + 1j * nussmall).flatten())
        nussmall = (np.arange(cols+1/mult-1/(mult*10),step=1/mult)*side).reshape(-1,1)
        nusbig = np.arange(rows+1,step=1)*side
        nussmall = nussmall - nussmall.mean()
        nusbig = nusbig - nusbig.mean()
        nus_all.append((nussmall  + 1j * nusbig).flatten())
        nus = np.hstack(nus_all).flatten()        
        nus = np.unique(nus)
        B = len(nus)

    elif topology_type==3:
        # Setup hexagonal topology
        nus = generate_hexagon_grid_corners(side, rows=rows, cols=cols)
        zone_centers = np.hstack([uniform_partition_hexagon(corners, n=1) for corners in nus])

    else:
        raise NotImplementedError("Selected topology_type is not implemented.")
    
    nus = np.unique(nus.flatten())
    U = len(zone_centers)
    B = len(nus)
    nus = np.unique((np.round(100*nus.real) + 1j* np.round(100*nus.imag))/100)
    B = nus.shape[0]
    
    # Add jitter if specified
    if jitter!=0:
        jitter_x = np.random.uniform(-jitter, jitter, B)
        jitter_y = np.random.uniform(-jitter, jitter, B)
        nus += (jitter_x + 1j * jitter_y)

    return U, B, zone_centers, nus

def plot_topology(nus, side, zone_centers, user_positions, topology_type=2, rows=3, cols=3, Qs=None):
    """
    Plot the system topology for different types of topologies.

    Parameters:
    nus : np.ndarray
        Array of user positions (complex numbers).
    side : float
        Side length of the zones or hexagons.
    zone_centers : np.ndarray
        Complex positions of zone centers.
    user_positions : list
        List of user positions for each zone or hexagon.
    topology_type : int
        Type of topology (2: rectangular grid, 3: hexagonal grid).
    rows : int, optional
        Number of rows in the grid for type-3 topology (default: 3).
    cols : int, optional
        Number of columns in the grid for type-3 topology (default: 3).
    Qs : np.ndarray, optional
        Array of additional positions to plot (e.g., codebook positions).

    Returns:
    None
    """
    if topology_type == 3:
        # Plot for hexagonal grid
        plot_hexagon_topology(
            generate_hexagon_grid_corners(side, rows=rows, cols=cols),
            user_positions,
            zone_centers
        )
    else:
        print("System topology where APs in orange dots, users in blue crosses, zone centers in green dots:")

        # Plot for rectangular grid or other topologies
        plt.figure(figsize=(6, 6))

        # Plot optional positions (e.g., Qs)
        if Qs is not None:
            plt.scatter(Qs.real, Qs.imag, color="black", s=10 if len(Qs) <= 2**6 else 1)

        # Plot zone boundaries and user positions
        for center in zone_centers:
            # Draw the zone boundaries
            plt.vlines(-0.5 * side + center.real, -0.5 * side + center.imag, 0.5 * side + center.imag, color="black", linewidth=1)
            plt.vlines(0.5 * side + center.real, -0.5 * side + center.imag, 0.5 * side + center.imag, color="black", linewidth=1)
            plt.hlines(-0.5 * side + center.imag, -0.5 * side + center.real, 0.5 * side + center.real, color="black", linewidth=1)
            plt.hlines(0.5 * side + center.imag, -0.5 * side + center.real, 0.5 * side + center.real, color="black", linewidth=1)

        # Plot user positions
        plt.scatter(user_positions.real, user_positions.imag, s=50, marker="x")

        # Plot zone centers
        plt.scatter(zone_centers.real, zone_centers.imag, s=30, color="green")

        # Plot all user positions
        plt.scatter(nus.real, nus.imag, marker="o", color="orange", s=200)

        # Customize plot appearance
        plt.ticklabel_format(style='sci', scilimits=(-side / 2, side / 2), axis='both')
        plt.axis("off")
        plt.title("System Topology")
        plt.tight_layout()
        plt.show()
