"""
Helper Functions for Cell-Free Massive MIMO System Setup

This file contains functions for simulating the TUMA framework in a cell-free massive MIMO setup.
It includes methods for generating system states, message types, sensor positions, and multiplicity
vectors, as well as computing received signals.

Functions:
- `gamma`: Computes large-scale fading coefficients based on Gkiouzepi and Caire's model.
- `qam_mod`: Maps quantized indices to quantized values.
- `qam_demod`: Maps quantized values to quantized indices.
- `generate_states_messages_types`: Generates active messages and their types for all zones.
- `generate_multiplicity_sensor_positions`: Assigns sensor positions and computes the multiplicity vector.
- `extract_user_positions`: Extracts user positions for visualization or processing.
- `generate_X`: Computes the \( \mathbf{X} \) matrix with fading channel effects.
- `transmit`: Simulates the transmission process, including signal generation and received signal computation.

Usage:
This module is intended for use in the TUMA framework and can be imported as:
    from utils.helper_cf_system import transmit, gamma

Author: Kaan Okumus
Date: January 2025
"""

import numpy as np


def gamma(q, v=0.0 + 1j * 0.0, rho=3.67, d0=0.01357):
    """
    Compute large-scale fading coefficients based on distance.

    Parameters:
    q : complex
        Position of the target user.
    v : complex, optional
        Reference point or AP position (default: origin).
    rho : float, optional
        Path loss exponent (default: 3.67).
    d0 : float, optional
        Reference distance (default: 0.01357).

    Returns:
    float
        Large-scale fading coefficient.
    """
    return 1 / (1 + (np.abs(q - v) / d0) ** rho)

def qam_mod(X, M, side):
    """
    Map quantized indices to quantized values.

    Parameters:
    X : np.ndarray
        Array of quantized indices.
    M : int
        Number of quantization levels.
    side : float
        Side length of the quantization region.

    Returns:
    np.ndarray
        Array of quantized values corresponding to the indices.
    """
    qs_x = -np.linspace(-side / 2 + side / int(np.sqrt(M)) / 2, side / 2 - side / int(np.sqrt(M)) / 2, int(np.sqrt(M)))
    qs_x, qs_y = (arr.ravel() for arr in np.meshgrid(qs_x, qs_x))
    qs = qs_x + 1j * qs_y 
    return qs[X]

def qam_demod(Y, M, side):
    """
    Map quantized values to quantized indices.

    Parameters:
    Y : np.ndarray
        Array of values to quantize.
    M : int
        Number of quantization levels.
    side : float
        Side length of the quantization region.

    Returns:
    np.ndarray
        Indices of the quantized values.
    """
    qs_x = -np.linspace(-side / 2 + side / int(np.sqrt(M)) / 2, side / 2 - side / int(np.sqrt(M)) / 2, int(np.sqrt(M)))
    qs_x, qs_y = (arr.ravel() for arr in np.meshgrid(qs_x, qs_x))
    qs = qs_x + 1j * qs_y
    return np.argmin(np.abs(qs - Y), axis=1)

def generate_states_messages_types(zones_infos, print_info=False, tx_types=None, force_Kmax=3):
    """
    Generate active messages and their transmission types for all zones.

    This function assigns target positions, active messages, and transmission types
    to each zone. It ensures unique active messages within each zone and enforces
    a maximum multiplicity limit (`force_Kmax`).

    Parameters:
    zones_infos : dict
        Information about each zone. Each zone entry should include:
        - `zone_center`: Center position of the zone (complex number).
        - `Mu`: Number of messages per zone.
        - `Mau`: Number of users in the zone.
        - `Kau`: Total number of transmitted messages in the zone.
        - `side`: Side length of the zone.
    print_info : bool, optional
        If `True`, prints detailed information about each zone (default: False).
    tx_types : list, optional
        Predefined transmission types for the zones. If `None`, types are randomly
        generated (default: None).
    force_Kmax : int, optional
        Maximum allowable multiplicity per message (default: 3).

    Returns:
    dict
        Updated `zones_infos` with the following additional keys for each zone:
        - `active_msg`: Array of active message indices.
        - `tx_type`: Array of transmission types for active messages.
        - `target_positions`: Complex positions of the target users in the zone.
    """
    for u in zones_infos.keys():
        zone_u_infos = zones_infos[u]
        zone_center = zone_u_infos["zone_center"]
        Mu = zone_u_infos["Mu"]
        Mau = zone_u_infos["Mau"]
        Kau = zone_u_infos["Kau"]
        side = zone_u_infos["side"]

        if print_info:
            print(f"Zone {u+1}: Mu = {Mu}, Mau = {Mau}, Kau = {Kau}")

        # Generate target positions within the zone
        target_positions = side * (-0.5 + np.random.rand(Mau, 1) + 1j * (-0.5 + np.random.rand(Mau, 1)))

        # Quantize positions to get active messages
        active_msg = qam_demod(target_positions, Mu, side)

        # Ensure unique active messages for each user
        while np.unique(active_msg).shape[0] != Mau:
            target_positions = side * (-0.5 + np.random.rand(Mau, 1) + 1j * (-0.5 + np.random.rand(Mau, 1)))
            active_msg = qam_demod(target_positions, Mu, side)

        # Generate or assign transmission types
        if tx_types is None:
            tx_type = np.random.multinomial(Kau, np.ones(Mau) / Mau)
            # Enforce multiplicity and active user constraints
            while (Mau != np.sum(tx_type != 0)) or (np.max(tx_type) > force_Kmax):
                tx_type = np.random.multinomial(Kau, np.ones(Mau) / Mau)
        else:
            tx_type = tx_types[u]

        if print_info:
            print(f"  Transmission types: {tx_type}")
            print(f"  Active messages: {active_msg}")

        # Align target positions to the zone center
        target_positions += zone_center

        # Update zone information
        zones_infos[u]["active_msg"] = active_msg
        zones_infos[u]["tx_type"] = tx_type
        zones_infos[u]["target_positions"] = target_positions

    return zones_infos

def generate_multiplicity_sensor_positions(zones_infos, print_info=False, spread_factor=1.0, 
                                           POS_MSG_DEPENDANCE=False, FORCE_POS_IN_GRIDS=True, margin=0.0):
    """
    Assign sensor positions and compute the multiplicity vector for all zones.

    This function assigns sensor positions for each active message based on specified constraints 
    (e.g., grid-based placement, proximity to access points) and computes the overall multiplicity 
    vector for all messages.

    Parameters:
    zones_infos : dict
        Information about each zone. Each zone entry should include:
        - `zone_center`: Center position of the zone (complex number).
        - `Mu`: Number of messages per zone.
        - `Mau`: Number of users in the zone.
        - `Kau`: Total number of transmitted messages in the zone.
        - `active_msg`: Array of active message indices.
        - `tx_type`: Array of transmission types for active messages.
        - `side`: Side length of the zone.
        - (Optional) `Qus`: Predefined grid points for message placement.
    print_info : bool, optional
        If `True`, prints detailed information about each zone (default: False).
    spread_factor : float, optional
        Scaling factor for random sensor placement (default: 1.0).
    POS_MSG_DEPENDANCE : bool, optional
        If `True`, sensor positions depend on target positions (default: False).
    FORCE_POS_IN_GRIDS : bool, optional
        If `True`, sensor positions are restricted to predefined grid points (default: True).
    margin : float, optional
        Margin to leave around the edges of the zone for sensor placement (default: 0.0).

    Returns:
    tuple
        zones_infos : dict
            Updated zone information with assigned sensor positions.
        k : np.ndarray
            Multiplicity vector for all zones and messages.
    """
    # Total number of messages across all zones
    M = sum([zones_infos[u]["Mu"] for u in zones_infos.keys()])

    # Initialize multiplicity vector
    k = np.zeros(M, dtype=int)

    for u in zones_infos.keys():
        zone_u_infos = zones_infos[u]
        zone_center = zone_u_infos["zone_center"]
        Mu = zone_u_infos["Mu"]
        Mau = zone_u_infos["Mau"]
        Kau = zone_u_infos["Kau"]
        side = zone_u_infos["side"]
        active_msg = zone_u_infos["active_msg"]
        tx_type = zone_u_infos["tx_type"]

        if print_info:
            print(f"Zone {u+1}:")
            print(f"  Active messages: {active_msg}")
            print(f"  Transmission types: {tx_type}")

        # Initialize sensor positions for the current zone
        sensor_positions = {msg: [] for msg in active_msg}

        # Optional grid constraints
        if FORCE_POS_IN_GRIDS:
            Qus = zone_u_infos["Qus"]
            chosen_grids = np.zeros(len(Qus), dtype=bool)

        # Assign multiplicities and sensor positions
        for ii, msg in enumerate(active_msg):
            k[msg + u * Mu] += tx_type[ii]  # Update multiplicity
            for _ in range(tx_type[ii]):  # Place multiple sensors for each type
                if POS_MSG_DEPENDANCE:
                    # Position depends on target positions
                    target_positions = zone_u_infos["target_positions"]
                    if FORCE_POS_IN_GRIDS:
                        raise NotImplementedError("POS_MSG_DEPENDANCE with FORCE_POS_IN_GRIDS is not implemented.")
                    else:
                        pos = (target_positions[ii] +
                               (np.random.rand() - 0.5 + 1j * (np.random.rand() - 0.5)) * side / (np.sqrt(Mu) * 2))
                else:
                    if FORCE_POS_IN_GRIDS:
                        # Select a grid point
                        chosen_grid_ind = np.random.choice(len(Qus))
                        while chosen_grids[chosen_grid_ind]:
                            chosen_grid_ind = np.random.choice(len(Qus))
                        chosen_grids[chosen_grid_ind] = True
                        pos = Qus[chosen_grid_ind]
                    else:
                        # Random position within zone bounds
                        pos = (zone_center +
                               spread_factor * (np.random.uniform(-side / 2 + margin, side / 2 - margin) +
                                                1j * np.random.uniform(-side / 2 + margin, side / 2 - margin)))

                sensor_positions[msg].append(pos)

        if print_info:
            print(f"  Sensor positions: {sensor_positions}")

        # Update zone information with sensor positions
        zones_infos[u]["sensor_positions"] = sensor_positions

        # Verify constraints
        ku = k[u * Mu:(u + 1) * Mu]
        assert Mau == len(ku[ku != 0])  # Ensure number of active messages matches
        assert Kau == ku.sum()  # Ensure total transmitted messages matches

    return zones_infos, k

def extract_user_positions(zones_infos, return_Qs=False):
    """
    Extract user positions from all zones for visualization or further processing.

    This function consolidates user positions across all zones into a single array. 
    Optionally, it can also return predefined grid points (\( \mathcal{Q}_u \)) 
    for each zone.

    Parameters:
    zones_infos : dict
        Information about each zone, including:
        - `sensor_positions`: Dictionary of sensor positions for each active message.
        - (Optional) `Qus`: Predefined grid points for each zone.
    return_Qs : bool, optional
        If `True`, returns predefined grid points along with user positions 
        (default: False).

    Returns:
    np.ndarray
        Array of all user positions across all zones.
    np.ndarray (optional)
        Array of predefined grid points (\( \mathcal{Q}_u \)) if `return_Qs=True`.
    """
    sensor_positions = []
    if return_Qs:
        Qs = []

    # Iterate through zones and collect sensor positions
    for u in zones_infos.keys():
        zone_u_infos = zones_infos[u]
        # Consolidate all sensor positions for the current zone
        sensor_positions.append(np.hstack(list(zone_u_infos["sensor_positions"].values())))
        if return_Qs and "Qus" in zone_u_infos:
            Qs.append(zone_u_infos["Qus"])

    # Stack all sensor positions into a single array
    sensor_positions = np.hstack(sensor_positions)

    if return_Qs:
        # Stack all grid points if requested
        Qs = np.hstack(Qs)
        return sensor_positions, Qs
    else:
        return sensor_positions

def generate_X(zones_infos, k, B, A, rho, d0, nus, print_info=False, include_AMP_DA=False, 
               perfect_CSI=True, imperfection_model="phase", sigma_noise_e=1, phase_max=np.pi/6):
    """
    Generate the \( \mathbf{X} \) matrix representing the fading channel-affected multiplicity vectors.

    Optionally, this function generates \( \mathbf{X}_{\text{for\_MDAircomp}} \) for AMP-DA simulation, 
    incorporating imperfect CSI if specified.

    Parameters:
    zones_infos : dict
        Information about each zone, including:
        - `Mu`: Number of messages in the zone.
        - `sensor_positions`: Positions of sensors corresponding to each active message.
    k : np.ndarray
        Multiplicity vector across all zones.
    B : int
        Number of access points (APs).
    A : int
        Number of antennas per AP.
    rho : float
        Path loss exponent.
    d0 : float
        Reference distance for path loss.
    nus : np.ndarray
        Positions of the APs.
    print_info : bool, optional
        If `True`, prints details about the generated \( \mathbf{X} \) (default: False).
    include_AMP_DA : bool, optional
        If `True`, generates \( \mathbf{X}_{\text{for\_MDAircomp}} \) for AMP-DA simulation (default: False).
    perfect_CSI : bool, optional
        If `True`, assumes perfect channel state information (CSI) (default: True).
    imperfection_model : str, optional
        Specifies the CSI imperfection model ("awgn" or "phase") (default: "phase").
    sigma_noise_e : float, optional
        Standard deviation of noise for imperfect CSI (default: 1).
    phase_max : float, optional
        Maximum phase error for the "phase" imperfection model (default: \( \pi / 6 \)).

    Returns:
    tuple
        X : list
            List of \( \mathbf{X}_u \) matrices for all zones, each of shape (\( M_u, F \)).
        X_for_MDAircomp : list (optional)
            List of \( \mathbf{X}_{\text{for\_MDAircomp}, u} \) matrices for AMP-DA simulation, if `include_AMP_DA=True`.
    """
    X = []
    if include_AMP_DA:
        X_for_MDAircomp = []

    F = B * A  # Total number of antennas across all APs

    for u in zones_infos.keys():
        zone_u_infos = zones_infos[u]
        Mu = zone_u_infos["Mu"]
        sensor_positions = zone_u_infos["sensor_positions"]
        ku = k[u * Mu:(u + 1) * Mu]  # Multiplicities for zone \( u \)

        # Initialize \( \mathbf{X}_u \) and optionally \( \mathbf{X}_{\text{for\_MDAircomp}, u} \)
        Xu = np.zeros((Mu, F), dtype=complex)
        if include_AMP_DA:
            Xu_for_MDAircomp = np.zeros((Mu, F), dtype=complex)

        for m, ku_m in enumerate(ku):
            if ku_m > 0:
                xu_m = np.zeros(F, dtype=complex)
                if include_AMP_DA:
                    xu_m_for_MDAircomp = np.zeros(F, dtype=complex)

                if print_info:
                    print(f"Zone {u+1}, Message {m}: Multiplicity = {ku_m}")

                for idxkum in range(ku_m):
                    # Sensor position for this transmission
                    pos = sensor_positions[m][idxkum]
                    
                    # Generate fading channel vector \( \mathbf{h} \)
                    h = (np.random.randn(F) + 1j * np.random.randn(F)) / np.sqrt(2)
                    h *= np.sqrt((gamma(pos, nus.reshape(-1, 1), rho=rho, d0=d0) * np.ones((1, A))).reshape(-1))
                    
                    # Update the transmitted signal vector \( \mathbf{x}_{u,m} \)
                    xu_m += h

                    if include_AMP_DA:
                        if perfect_CSI:
                            h_e = h[0]  # Use exact channel vector
                        else:
                            if imperfection_model == "phase":
                                phase = np.random.uniform(-phase_max, phase_max)
                                h_e = h[0] * np.exp(1j * phase)
                            elif imperfection_model == "awgn":
                                noise_e = (np.random.randn() + 1j * np.random.randn()) * (sigma_noise_e / np.sqrt(2))
                                h_e = h[0] + noise_e
                            else:
                                raise ValueError(f"Unknown imperfection_model: {imperfection_model}. Valid options: 'awgn', 'phase'.")
                        
                        xu_m_for_MDAircomp += h / h_e

                Xu[m] = xu_m
                if include_AMP_DA:
                    Xu_for_MDAircomp[m] = xu_m_for_MDAircomp

        X.append(Xu)
        if include_AMP_DA:
            X_for_MDAircomp.append(Xu_for_MDAircomp)

    if include_AMP_DA:
        return X, X_for_MDAircomp
    else:
        return X

def transmit(Mus, Maus, Kaus, side, n, F, A, U, Cx, nP, sigma_w, nus, zone_centers, 
             rho, d0, margin=0.0, include_AMP_DA=False, perfect_CSI=True, 
             imperfection_model="phase", sigma_noise_e=1, phase_max=np.pi/6, 
             tx_types=None, POS_MSG_DEPENDANCE=False, FORCE_POS_IN_GRIDS=False, 
             J_for_Qus=None, print_info=False, force_Kmax=3):
    """
    Simulates the transmission process in the TUMA framework with cell-free massive MIMO.

    Parameters:
    Mus : list[int]
        List of numbers of messages (\( M_u \)) per zone.
    Maus : list[int]
        List of numbers of users (\( M_{au} \)) per zone.
    Kaus : list[int]
        List of total transmitted messages (\( K_{au} \)) per zone.
    side : float
        Length of the side of the simulation area.
    n : int
        Block length.
    F : int
        Total number of antennas (\( B \times A \)).
    A : int
        Number of antennas per AP.
    U : int
        Number of zones.
    Cx : callable
        Function to compute \( \mathbf{C}_u \mathbf{X}_u \).
    nP : float
        Total transmission power per user.
    sigma_w : float
        Noise standard deviation.
    nus : np.ndarray
        Positions of the APs.
    zone_centers : np.ndarray
        Centers of the zones.
    rho : float
        Path loss exponent.
    d0 : float
        Reference distance for path loss.
    margin : float, optional
        Margin to leave around the edges of the zone for sensor placement (default: 0.0).
    include_AMP_DA : bool, optional
        If `True`, prepares data for AMP-DA simulation (default: False).
    perfect_CSI : bool, optional
        If `True`, assumes perfect channel state information (CSI) (default: True).
    imperfection_model : str, optional
        Specifies the CSI imperfection model ("awgn" or "phase") (default: "phase").
    sigma_noise_e : float, optional
        Standard deviation of noise for imperfect CSI (default: 1).
    phase_max : float, optional
        Maximum phase error for the "phase" imperfection model (default: \( \pi / 6 \)).
    tx_types : list[np.ndarray], optional
        Transmission types for each zone. If `None`, types are generated randomly (default: None).
    POS_MSG_DEPENDANCE : bool, optional
        If `True`, sensor positions depend on target positions (default: False).
    FORCE_POS_IN_GRIDS : bool, optional
        If `True`, sensor positions are restricted to predefined grid points (default: False).
    J_for_Qus : int, optional
        Number of quantization bits for grid points (\( J_u \)), required if `FORCE_POS_IN_GRIDS=True`.
    print_info : bool, optional
        If `True`, prints detailed information about the transmission process (default: False).
    force_Kmax : int, optional
        Maximum multiplicity per active message (default: 3).

    Returns:
    tuple
        Y : np.ndarray
            Received signal matrix.
        (Optional) Y_for_MDAircomp : np.ndarray
            Received signal matrix for AMP-DA simulation, if `include_AMP_DA=True`.
        k : np.ndarray
            Multiplicity vector across all zones.
        X : list
            List of \( \mathbf{X}_u \) matrices for all zones.
        (Optional) X_for_MDAircomp : list
            List of \( \mathbf{X}_{\text{for\_MDAircomp}, u} \) matrices, if `include_AMP_DA=True`.
        zones_infos : dict
            Updated zone information.
    """
    M = sum(Mus)  # Total number of messages
    B = len(nus)  # Total number of APs

    # Initialize zone information
    zones_infos = {}
    for u, zone_center in enumerate(zone_centers):
        zones_infos[u] = {
            "zone_center": zone_center,
            "Mu": Mus[u],
            "Mau": Maus[u],
            "Kau": Kaus[u],
            "side": side,
        }
        if FORCE_POS_IN_GRIDS:
            assert J_for_Qus is not None, "J_for_Qus must be provided if FORCE_POS_IN_GRIDS=True."
            M_for_Qus = 2**J_for_Qus
            qs_x = np.linspace(-side / 2, side / 2, int(np.sqrt(M_for_Qus)))
            qs_x, qs_y = np.meshgrid(qs_x, qs_x)
            zones_infos[u]["Qus"] = qs_x.ravel() + 1j * qs_y.ravel()

    # Generate states, messages, and types
    zones_infos = generate_states_messages_types(zones_infos, print_info=print_info, tx_types=tx_types, force_Kmax=force_Kmax)
    zones_infos, k = generate_multiplicity_sensor_positions(zones_infos, print_info=print_info, 
                                                            POS_MSG_DEPENDANCE=POS_MSG_DEPENDANCE, 
                                                            FORCE_POS_IN_GRIDS=FORCE_POS_IN_GRIDS, margin=margin)

    # Generate \( \mathbf{X} \) and \( \mathbf{X}_{\text{for\_MDAircomp}} \)
    if include_AMP_DA:
        X, X_for_MDAircomp = generate_X(zones_infos, k, B, A, rho, d0, nus, print_info=print_info, 
                                        include_AMP_DA=True, perfect_CSI=perfect_CSI, 
                                        imperfection_model=imperfection_model, sigma_noise_e=sigma_noise_e, 
                                        phase_max=phase_max)
    else:
        X = generate_X(zones_infos, k, B, A, rho, d0, nus, print_info=print_info)

    # Add noise to the received signal
    W = (np.random.randn(n, F) + 1j * np.random.randn(n, F)) * np.sqrt(sigma_w / 2)

    # Compute the received signal \( \mathbf{Y} \)
    Y = W.copy()
    for u in range(U):
        Y += np.sqrt(nP) * Cx(X[u], u)

    if include_AMP_DA:
        Y_for_MDAircomp = W.copy()
        for u in range(U):
            Y_for_MDAircomp += np.sqrt(nP) * Cx(X_for_MDAircomp[u], u)
        return Y, Y_for_MDAircomp, k, X, X_for_MDAircomp, zones_infos

    return Y, k, X, zones_infos
