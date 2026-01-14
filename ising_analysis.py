"""
Ising Model Simulation in Heat Bath

This module implements a Monte Carlo simulation of the 2D Ising model
using a heat bath algorithm. The simulation includes:
- Random initialization
- Burn-in phase
- Two measurement blocks for convergence checking
- Energy and magnetization calculations
"""

import random
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from datetime import datetime
import os


# ===== Numba-optimized core functions =====

@njit
def sweep_numba(lattice, eta, meuB, L, E_now, E_flip, sum_of_nn):
    """
    Numba-optimized sweep function.
    
    Parameters:
    -----------
    lattice : ndarray
        2D array of spins
    eta : float
        Spin coupling constant
    meuB : float
        External field parameter
    L : int
        Lattice size
    E_now : ndarray
        Pre-allocated array for current energies
    E_flip : ndarray
        Pre-allocated array for flipped energies
    sum_of_nn : ndarray
        Pre-allocated array for nearest neighbor sums
        
    Returns:
    --------
    int
        Number of spins that flipped
    """
    flips = 0
    
    for i in range(L):
        for j in range(L):
            spin_i = lattice[i, j]
            
            # Periodic boundary conditions
            up = lattice[(i - 1) % L, j]
            down = lattice[(i + 1) % L, j]
            left = lattice[i, (j - 1) % L]
            right = lattice[i, (j + 1) % L]
            
            # Sum of nearest neighbors weighted by current spin
            sum_nn = spin_i * (up + down + left + right)
            sum_of_nn[i, j] = sum_nn
            
            # Energy of current state and flipped state
            E_now[i, j] = -eta * sum_nn - meuB * spin_i
            E_flip[i, j] = eta * sum_nn + meuB * spin_i
            
            # Heat bath probability: P(flip) = exp(-E_flip) / (exp(-E_flip) + exp(-E_now))
            exp_flip = np.exp(-E_flip[i, j])
            exp_now = np.exp(-E_now[i, j])
            P_i_flip = exp_flip / (exp_flip + exp_now)
            
            # Randomly decide to flip based on probability
            if np.random.random() <= P_i_flip:
                lattice[i, j] = -lattice[i, j]
                flips += 1
    
    return flips


@njit
def magnetization_numba(lattice):
    """
    Numba-optimized magnetization calculation.
    
    Parameters:
    -----------
    lattice : ndarray
        2D array of spins
        
    Returns:
    --------
    float
        Sum of all spins
    """
    total = 0.0
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            total += lattice[i, j]
    return total


@njit
def energy_numba(lattice, eta, meuB, L):
    """
    Numba-optimized energy calculation.
    
    Parameters:
    -----------
    lattice : ndarray
        2D array of spins
    eta : float
        Spin coupling constant
    meuB : float
        External field parameter
    L : int
        Lattice size
        
    Returns:
    --------
    float
        Total energy
    """
    E_int = 0.0
    E_field = 0.0
    
    for i in range(L):
        for j in range(L):
            spin_i = lattice[i, j]
            E_field += spin_i
            
            # Sum of nearest neighbors
            up = lattice[(i - 1) % L, j]
            down = lattice[(i + 1) % L, j]
            left = lattice[i, (j - 1) % L]
            right = lattice[i, (j + 1) % L]
            
            nn_sum = up + down + left + right
            E_int += spin_i * nn_sum
    
    E_int = -0.5 * eta * E_int
    E_field = -meuB * E_field
    
    return E_int + E_field


# ===== Global Parameters =====
L = 32                    # Lattice size (L x L)
eta = 0.6                 # Spin coupling constant J
meuB = 0                  # External magnetic field parameter
nsweep = 5                # Number of sweeps between measurements
K = 80000                 # Total number of spin flips per block
kB = 1.380649e-23         # Boltzmann constant
Delta = 1e-3              # Convergence threshold for relative difference


class IsingLattice:
    """
    Represents a 2D Ising model lattice with Monte Carlo dynamics.
    
    The lattice is initialized randomly with spins ±1, and evolves
    through heat bath updates where each spin flips with probability
    based on its local energy.
    """
    
    def __init__(self, L, eta, meuB):
        """
        Initialize the Ising lattice.
        
        Parameters:
        -----------
        L : int
            Lattice size (L x L)
        eta : float
            Spin coupling constant J
        meuB : float
            External magnetic field parameter
        """
        self.L = L
        self.eta = eta
        self.meuB = meuB
        self.N = L * L
        
        # Initialize random lattice with spins ±1
        rand_bits = np.random.randint(0, 2, size=(L, L))
        self.lattice = 2 * rand_bits - 1  # 0 -> -1, 1 -> +1
        
        # Pre-allocate arrays for energy calculations
        self.E_now = np.zeros((L, L))
        self.E_flip = np.zeros((L, L))
        self.sum_of_nn = np.zeros((L, L))
    
    def sweep(self):
        """
        Perform one sweep over the entire lattice.
        
        For each spin, calculate the probability of flipping based on
        nearest neighbors and the heat bath algorithm, then randomly
        decide whether to flip.
        
        Returns:
        --------
        int
            Number of spins that actually flipped in this sweep
        """
        return sweep_numba(self.lattice, self.eta, self.meuB, self.L,
                          self.E_now, self.E_flip, self.sum_of_nn)
    
    def magnetization(self):
        """
        Calculate the total magnetization of the lattice.
        
        Returns:
        --------
        float
            Sum of all spins in the lattice
        """
        return magnetization_numba(self.lattice)
    
    def energy(self):
        """
        Calculate the total energy of the lattice.
        
        The energy consists of:
        - Interaction energy: -0.5 * eta * sum(spin_i * sum(neighbors))
        - Field energy: -meuB * sum(spins)
        
        Returns:
        --------
        float
            Total energy of the lattice
        """
        return energy_numba(self.lattice, self.eta, self.meuB, self.L)


def burn_in(lattice, n_burn):
    """
    Perform burn-in phase to forget initial random state.
    
    Parameters:
    -----------
    lattice : IsingLattice
        The lattice to evolve
    n_burn : int
        Target number of spin flips for burn-in
        
    Returns:
    --------
    tuple
        (total_number_of_flips, number_of_sweeps)
    """
    num_flips = 0
    num_sweeps = 0
    while num_flips < n_burn:
        num_flips += lattice.sweep()
        num_sweeps += 1
    return num_flips, num_sweeps


def measure_block1(lattice, num_flips, target_1, nsweep):
    """
    First measurement block: measure magnetization every nsweep sweeps.
    
    Parameters:
    -----------
    lattice : IsingLattice
        The lattice to measure
    num_flips : int
        Current number of flips (continues from burn-in)
    target_1 : int
        Target number of flips to reach (end of block 1)
    nsweep : int
        Number of sweeps between measurements
        
    Returns:
    --------
    tuple
        (updated_num_flips, list_of_magnetization_values, number_of_sweeps)
    """
    M_block1 = []
    num_sweeps = 0
    
    while num_flips < target_1:
        # Perform nsweep sweeps between measurements
        for _ in range(nsweep):
            num_flips += lattice.sweep()
            num_sweeps += 1
            if num_flips >= target_1:
                break
        M_block1.append(lattice.magnetization())
    
    return num_flips, M_block1, num_sweeps


def measure_block2(lattice, num_flips, target_2, nsweep):
    """
    Second measurement block: measure both magnetization and energy.
    
    Parameters:
    -----------
    lattice : IsingLattice
        The lattice to measure
    num_flips : int
        Current number of flips (continues from block 1)
    target_2 : int
        Target number of flips to reach (end of block 2)
    nsweep : int
        Number of sweeps between measurements
        
    Returns:
    --------
    tuple
        (updated_num_flips, list_of_magnetization_values, list_of_energy_values, number_of_sweeps)
    """
    M_block2 = []
    E_block2 = []
    num_sweeps = 0
    
    while num_flips < target_2:
        # Perform one sweep before measurement
        num_flips += lattice.sweep()
        num_sweeps += 1
        
        # Then perform nsweep sweeps between measurements
        for _ in range(nsweep):
            num_flips += lattice.sweep()
            num_sweeps += 1
            if num_flips >= target_2:
                break
        
        M_block2.append(lattice.magnetization())
        E_block2.append(lattice.energy())
    
    return num_flips, M_block2, E_block2, num_sweeps


def check_convergence(M_half_mean, M_K_mean, Delta):
    """
    Check if magnetization has converged based on relative difference.
    
    Convergence criterion: |M_K - M_half| / |M_K| <= Delta
    
    Parameters:
    -----------
    M_half_mean : float
        Mean magnetization from first block
    M_K_mean : float
        Mean magnetization from second block
    Delta : float
        Convergence threshold
        
    Returns:
    --------
    tuple
        (converged: bool, relative_difference: float)
    """
    relative_diff = abs(M_K_mean - M_half_mean) / abs(M_K_mean)
    converged = relative_diff <= Delta
    return converged, relative_diff


def calculate_heat_capacity(E_block2, kB, N):
    """
    Calculate heat capacity from energy fluctuations.
    
    Cv = (kB/N) * (⟨E²⟩ - ⟨E⟩²)
    
    Parameters:
    -----------
    E_block2 : array-like
        List of energy measurements from block 2
    kB : float
        Boltzmann constant
    N : int
        Number of spins
        
    Returns:
    --------
    tuple
        (mean_energy, heat_capacity)
    """
    E_array = np.array(E_block2, dtype=float)
    E_mean = np.mean(E_array)
    E2_mean = np.mean(E_array ** 2)
    Cv = (kB / N) * (E2_mean - E_mean ** 2)
    return E_mean, Cv


def print_results(M_half_mean, M_K_mean, relative_diff, converged,
                  E_mean, Cv, L):
    """
    Print simulation results in a formatted way.
    
    Parameters:
    -----------
    M_half_mean : float
        Mean magnetization from first block
    M_K_mean : float
        Mean magnetization from second block
    relative_diff : float
        Relative difference between blocks
    converged : bool
        Whether convergence was achieved
    E_mean : float
        Mean energy from block 2
    Cv : float
        Heat capacity
    L : int
        Lattice size
    """
    print("=" * 50)
    print("Ising Model Simulation Results")
    print("=" * 50)
    print(f"⟨M⟩_{{K/2}} = {M_half_mean:.6f}")
    print(f"⟨M⟩_{{K}}   = {M_K_mean:.6f}")
    print(f"Relative difference = {relative_diff:.6e}")
    print(f"Converged? {converged}")
    print()
    print(f"⟨E⟩_block2 = {E_mean:.6f}")
    print(f"Cv         = {Cv:.6e}")
    print(f"⟨M⟩_K per spin = {M_K_mean / (L * L):.6f}")
    print(f"⟨E⟩ per spin    = {E_mean / (L * L):.6f}")
    print("=" * 50)


def run_ising_simulation(eta, meuB=0, L=32, K=80000, nsweep=5, Delta=1e-3, kB=1.380649e-23, max_flips=1e8, verbose=True):
    """
    Run a complete Ising model simulation for given parameters.
    
    The simulation proceeds in phases:
    1. Burn-in: K/2 flips to forget initial state
    2. Block 1: K/2 flips measuring magnetization (⟨M⟩_{K/2})
    3. Block 2: K flips measuring magnetization and energy (⟨M⟩_K)
    4. If convergence fails, iteratively run additional Block 2 phases
       with doubled sizes (2K, 4K, 8K, ...) until convergence is achieved.
    
    Each iteration compares the new M_K_mean to the previous one.
    If the total number of flips exceeds max_flips, the simulation stops
    and treats the result as converged.
    
    Parameters:
    -----------
    eta : float
        Spin coupling constant J
    meuB : float, optional
        External magnetic field parameter (default: 0)
    L : int, optional
        Lattice size (default: 32)
    K : int, optional
        Total number of spin flips per block (default: 80000)
    nsweep : int, optional
        Number of sweeps between measurements (default: 5)
    Delta : float, optional
        Convergence threshold (default: 1e-3)
    kB : float, optional
        Boltzmann constant (default: 1.380649e-23)
    max_flips : float, optional
        Maximum number of flips before forcing convergence (default: 1e8)
    verbose : bool, optional
        Whether to print progress messages (default: True)
        
    Returns:
    --------
    dict
        Dictionary containing all simulation results and statistics
    """
    # Calculate derived parameters
    N = L * L
    n_burn = K // 2              # Burn-in: first K/2 flips
    target_1 = K                 # End of first measurement block
    
    # Initialize lattice
    lattice = IsingLattice(L, eta, meuB)
    
    # Phase 1: Burn-in
    if verbose:
        print(f"  Phase 1: Burn-in...")
    num_flips, sweeps_burn = burn_in(lattice, n_burn)
    total_sweeps = sweeps_burn
    if verbose:
        print(f"    Completed {num_flips} flips")
    
    # Phase 2: First measurement block
    if verbose:
        print(f"  Phase 2: First measurement block...")
    num_flips, M_block1, sweeps_block1 = measure_block1(lattice, num_flips, target_1, nsweep)
    total_sweeps += sweeps_block1
    M_half_mean = np.mean(M_block1)
    if verbose:
        print(f"    Completed {num_flips} flips, {len(M_block1)} measurements")
    
    # Phase 3: Initial second measurement block (K flips)
    block_size = K
    target_current = num_flips + block_size
    if verbose:
        print(f"  Phase 3: Second measurement block ({block_size} flips)...")
    num_flips, M_block2, E_block2, sweeps_block2 = measure_block2(lattice, num_flips, target_current, nsweep)
    total_sweeps += sweeps_block2
    M_K_mean = np.mean(M_block2)
    if verbose:
        print(f"    Completed {num_flips} flips, {len(M_block2)} measurements")
    
    # Check initial convergence
    converged, relative_diff = check_convergence(M_half_mean, M_K_mean, Delta)
    
    # Iterative convergence: if not converged, run additional blocks with doubled sizes
    iteration = 1
    previous_M_K_mean = M_K_mean
    all_M_block2 = M_block2.copy()
    all_E_block2 = E_block2.copy()
    forced_convergence = False
    
    while not converged:
        iteration += 1
        block_size *= 2  # Double the block size
        target_current = num_flips + block_size
        
        if verbose:
            print(f"  Phase 3 (iteration {iteration}): Additional measurement block ({block_size} flips)...")
        num_flips, M_block2_new, E_block2_new, sweeps_new = measure_block2(lattice, num_flips, target_current, nsweep)
        total_sweeps += sweeps_new
        M_K_mean_new = np.mean(M_block2_new)
        if verbose:
            print(f"    Completed {num_flips} flips, {len(M_block2_new)} measurements")
        
        # Check convergence relative to previous M_K_mean
        converged, relative_diff = check_convergence(previous_M_K_mean, M_K_mean_new, Delta)
        
        # Accumulate measurements for final statistics
        all_M_block2.extend(M_block2_new)
        all_E_block2.extend(E_block2_new)
        
        # Update for next iteration
        previous_M_K_mean = M_K_mean_new
        M_K_mean = M_K_mean_new
        
        # Check if we've exceeded max_flips after completing this iteration
        # If so, don't start the next iteration
        if num_flips >= max_flips:
            if verbose:
                print(f"    Maximum flip limit reached ({max_flips:.0e} flips). Stopping before next iteration.")
            converged = True
            forced_convergence = True
        
        if verbose:
            if converged:
                if forced_convergence:
                    print(f"    Convergence forced after {iteration} iterations (max flips limit reached)!")
                else:
                    print(f"    Convergence achieved after {iteration} iterations!")
            else:
                print(f"    Convergence not yet achieved (relative diff: {relative_diff:.6e})")
    
    # Calculate final statistics using all accumulated measurements
    M_K_mean_final = np.mean(all_M_block2)
    E_mean, Cv = calculate_heat_capacity(all_E_block2, kB, N)
    
    # Calculate flip ratio: flips / (sweeps * N)
    flip_attempts = total_sweeps * N
    flip_ratio = num_flips / flip_attempts if flip_attempts > 0 else 0.0
    
    # Print results if verbose
    if verbose:
        print_results(M_half_mean, M_K_mean_final, relative_diff, converged,
                      E_mean, Cv, L)
    
    # Return results for potential further analysis
    return {
        'eta': eta,
        'meuB': meuB,
        'L': L,
        'M_half_mean': M_half_mean,
        'M_K_mean': M_K_mean_final,
        'relative_diff': relative_diff,
        'converged': converged,
        'forced_convergence': forced_convergence,
        'E_mean': E_mean,
        'Cv': Cv,
        'M_block1': M_block1,
        'M_block2': all_M_block2,
        'E_block2': all_E_block2,
        'total_iterations': iteration,
        'total_flips': num_flips,
        'total_sweeps': total_sweeps,
        'flip_ratio': flip_ratio
    }


def main(eta_min=0.1, eta_max=0.8, eta_step=0.05, add_fine_eta_window=False, meuB_values=None):
    """
    Main function that runs Ising model simulations for multiple eta and meuB values.
    
    Parameters:
    -----------
    eta_min : float, optional
        Minimum eta value (default: 0.1)
    eta_max : float, optional
        Maximum eta value (default: 0.8)
    eta_step : float, optional
        Step size for eta values (default: 0.05)
    add_fine_eta_window : bool, optional
        If True, adds a fine-grained eta window from 0.42 to 0.46 with step 0.005
        (default: False)
    meuB_values : list, optional
        List of meuB values to run simulations for. If None, uses the global meuB value.
        (default: None, which means [0] if not specified, or [0.1, 0.5, 1.0] if specified)
    
    Returns:
    --------
    list
        List of all simulation results
    """
    # Generate main eta values and reverse to start from highest
    eta_values = np.arange(eta_min, eta_max + eta_step/2, eta_step)
    
    # Add fine-grained window if requested
    if add_fine_eta_window:
        eta_fine_min = 0.42
        eta_fine_max = 0.46
        eta_fine_step = 0.005
        eta_fine_values = np.arange(eta_fine_min, eta_fine_max + eta_fine_step/2, eta_fine_step)
        # Combine and remove duplicates, then sort descending
        eta_values = np.concatenate([eta_values, eta_fine_values])
        eta_values = np.unique(eta_values)
        eta_fine_info = f"\nFine eta window: {eta_fine_min} to {eta_fine_max} (step: {eta_fine_step})"
    else:
        eta_fine_info = ""
    
    # Set meuB values
    if meuB_values is None:
        meuB_values = [meuB]  # Use global meuB value
    elif not isinstance(meuB_values, (list, np.ndarray)):
        meuB_values = [meuB_values]
    
    meuB_values = np.array(meuB_values)
    
    print("=" * 70)
    print("Running Ising Model Simulations")
    print("=" * 70)
    print(f"Eta range: {eta_min} to {eta_max} (step: {eta_step}){eta_fine_info}")
    print(f"Running from highest to lowest eta")
    print(f"meuB values: {meuB_values}")
    print(f"Total eta values: {len(eta_values)}")
    print(f"Total meuB values: {len(meuB_values)}")
    print(f"Total simulations: {len(eta_values) * len(meuB_values)}")
    print(f"Maximum flips per simulation: {1e8:.0e}")
    print("=" * 70)
    print()
    
    # Store results for all simulations
    all_results = []
    
    # Run simulation for each meuB value
    for meuB_idx, meuB_val in enumerate(meuB_values, 1):
        print(f"\n{'=' * 70}")
        print(f"meuB = {meuB_val:.2f} ({meuB_idx}/{len(meuB_values)})")
        print("=" * 70)
        
        # Run simulation for each eta value
        for eta_idx, eta_val in enumerate(eta_values, 1):
            sim_num = (meuB_idx - 1) * len(eta_values) + eta_idx
            total_sims = len(eta_values) * len(meuB_values)
            
            print(f"\n[{sim_num}/{total_sims}] Running simulation for eta = {eta_val:.4f}, meuB = {meuB_val:.2f}")
            print("-" * 70)
            
            # Run simulation with verbose output
            results = run_ising_simulation(
                eta=eta_val,
                meuB=meuB_val,
                L=L,
                K=K,
                nsweep=nsweep,
                Delta=Delta,
                kB=kB,
                max_flips=1e8,
                verbose=True
            )
            
            all_results.append(results)
            print()
    
    # Print summary of all results
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL SIMULATIONS")
    print("=" * 70)
    print(f"{'meuB':<8} {'Eta':<10} {'⟨M⟩_K':<12} {'⟨E⟩':<12} {'Cv':<15} {'Converged':<12} {'Forced':<10} {'Iterations':<12}")
    print("-" * 70)
    
    for res in all_results:
        converged_str = str(res['converged'])
        forced_str = str(res.get('forced_convergence', False))
        print(f"{res['meuB']:<8.2f} {res['eta']:<10.4f} {res['M_K_mean']:<12.2f} {res['E_mean']:<12.2f} "
              f"{res['Cv']:<15.6e} {converged_str:<12} {forced_str:<10} {res['total_iterations']:<12}")
    
    print("=" * 70)
    
    return all_results


def onsager_magnetization(eta, eta_c=0.44068):
    """
    Calculate the Onsager solution for magnetization per particle.
    
    M/N = ±[(1+z²)^(1/4) * (1-6z²+z⁴)^(1/8)] / (1-z²)^(1/2)
    where z = exp(-2η)
    for η > η_c
    
    Parameters:
    -----------
    eta : array-like
        Eta values
    eta_c : float
        Critical eta value (default: 0.44068)
        
    Returns:
    --------
    array
        Magnetization per particle (absolute value)
    """
    eta = np.array(eta)
    result = np.zeros_like(eta)
    
    # Only calculate for eta > eta_c
    mask = eta > eta_c
    if np.any(mask):
        z = np.exp(-2 * eta[mask])
        z2 = z**2
        z4 = z**4
        
        numerator = (1 + z2)**(1/4) * (1 - 6*z2 + z4)**(1/8)
        denominator = (1 - z2)**(1/2)
        
        # Take absolute value
        result[mask] = np.abs(numerator / denominator)
    
    return result


def plot_results(all_results, L=32):
    """
    Create visualization plots from simulation results.
    
    Creates 6 separate graphs saved in a timestamped results folder:
    1. |M|/N vs eta for meuB=0 (with Onsager comparison)
    2. <E>/N vs eta for meuB=0
    3. Cv vs eta for meuB=0
    4. Flip ratio (flips/attempts) vs eta for meuB=0
    5. M/N vs eta for all meuB (color-coded, |M|/N for meuB=0)
    6. <E>/N vs eta for all meuB (color-coded)
    
    Parameters:
    -----------
    all_results : list
        List of simulation result dictionaries
    L : int
        Lattice size (for calculating N)
    """
    # Create timestamped results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nSaving results to folder: {results_dir}/")
    
    N = L * L
    
    # Separate results by meuB
    results_by_meuB = {}
    for res in all_results:
        meuB_val = res['meuB']
        if meuB_val not in results_by_meuB:
            results_by_meuB[meuB_val] = []
        results_by_meuB[meuB_val].append(res)
    
    # Sort each meuB group by eta
    for meuB_val in results_by_meuB:
        results_by_meuB[meuB_val].sort(key=lambda x: x['eta'])
    
    # Get meuB=0 results
    meuB_0_results = results_by_meuB.get(0, [])
    
    # Prepare data for meuB=0
    if meuB_0_results:
        eta_0 = np.array([r['eta'] for r in meuB_0_results])
        M_abs_0 = np.abs(np.array([r['M_K_mean'] for r in meuB_0_results])) / N
        E_0 = np.array([r['E_mean'] for r in meuB_0_results]) / N
        Cv_0 = np.array([r['Cv'] for r in meuB_0_results])
        flip_ratio_0 = np.array([r['flip_ratio'] for r in meuB_0_results])
    
    # Graph 1: |M|/N vs eta for meuB=0 with Onsager comparison
    if meuB_0_results:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(eta_0, M_abs_0, 'o-', label='Simulation', linewidth=2, markersize=6)
        
        # Onsager solution
        eta_onsager = np.linspace(0.1, 0.8, 200)
        M_onsager = onsager_magnetization(eta_onsager)
        ax1.plot(eta_onsager, M_onsager, '--', label='Onsager Solution', linewidth=2, color='red')
        
        ax1.axvline(x=0.44068, color='gray', linestyle=':', linewidth=1, label='η_c = 0.44068')
        ax1.set_xlabel('η', fontsize=12)
        ax1.set_ylabel('|M|/N', fontsize=12)
        ax1.set_title('Absolute Mean Magnetization per Particle (h=0)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/1_magnetization_h0_with_onsager.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 1_magnetization_h0_with_onsager.png")
    
    # Graph 2: <E>/N vs eta for meuB=0
    if meuB_0_results:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(eta_0, E_0, 'o-', linewidth=2, markersize=6, color='green')
        ax2.set_xlabel('η', fontsize=12)
        ax2.set_ylabel('<E>/N', fontsize=12)
        ax2.set_title('Mean Energy per Particle (h=0)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0.44068, color='gray', linestyle=':', linewidth=1)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/2_energy_h0.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 2_energy_h0.png")
    
    # Graph 3: Cv vs eta for meuB=0
    if meuB_0_results:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(eta_0, Cv_0, 'o-', linewidth=2, markersize=6, color='purple')
        ax3.set_xlabel('η', fontsize=12)
        ax3.set_ylabel('Cv', fontsize=12)
        ax3.set_title('Heat Capacity (h=0)', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0.44068, color='gray', linestyle=':', linewidth=1)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/3_heat_capacity_h0.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 3_heat_capacity_h0.png")
    
    # Graph 4: Flip ratio (flips/attempts) vs eta for meuB=0
    # Include eta=2.0 if it exists in results
    if meuB_0_results:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        # Get all meuB=0 results (including eta=2.0 if present)
        all_meuB_0_results = [r for r in all_results if r['meuB'] == 0]
        all_meuB_0_results.sort(key=lambda x: x['eta'])
        
        eta_all_0 = np.array([r['eta'] for r in all_meuB_0_results])
        flip_ratio_all_0 = np.array([r['flip_ratio'] for r in all_meuB_0_results])
        
        # Plot regular points
        regular_mask = eta_all_0 <= 0.8
        if np.any(regular_mask):
            ax4.plot(eta_all_0[regular_mask], flip_ratio_all_0[regular_mask], 
                    'o-', linewidth=2, markersize=6, color='orange', label='Regular range')
        
        # Plot eta=2.0 point separately if it exists
        eta_2_mask = eta_all_0 == 2.0
        if np.any(eta_2_mask):
            ax4.plot(eta_all_0[eta_2_mask], flip_ratio_all_0[eta_2_mask], 
                    's', markersize=10, color='red', label='η=2.0', zorder=5)
        
        ax4.set_xlabel('η', fontsize=12)
        ax4.set_ylabel('Flip Ratio (Flips/Attempts)', fontsize=12)
        ax4.set_title('Spin Flip Acceptance Ratio (h=0)', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0.44068, color='gray', linestyle=':', linewidth=1)
        if np.any(eta_2_mask):
            ax4.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/4_flip_ratio_h0.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 4_flip_ratio_h0.png")
    
    # Graph 5: M/N vs eta for all meuB (color-coded)
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_by_meuB)))
    for (meuB_val, results), color in zip(sorted(results_by_meuB.items()), colors):
        eta_vals = np.array([r['eta'] for r in results])
        if meuB_val == 0:
            # Use absolute value for meuB=0
            M_vals = np.abs(np.array([r['M_K_mean'] for r in results])) / N
        else:
            M_vals = np.array([r['M_K_mean'] for r in results]) / N
        
        ax5.plot(eta_vals, M_vals, 'o-', label=f'h={meuB_val:.1f}', 
                linewidth=2, markersize=6, color=color)
    
    ax5.set_xlabel('η', fontsize=12)
    ax5.set_ylabel('M/N', fontsize=12)
    ax5.set_title('Mean Magnetization per Particle (All h)', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10, title='h (meuB)')
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/5_magnetization_all_h.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 5_magnetization_all_h.png")
    
    # Graph 6: <E>/N vs eta for all meuB (color-coded)
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    for (meuB_val, results), color in zip(sorted(results_by_meuB.items()), colors):
        eta_vals = np.array([r['eta'] for r in results])
        E_vals = np.array([r['E_mean'] for r in results]) / N
        
        ax6.plot(eta_vals, E_vals, 'o-', label=f'h={meuB_val:.1f}', 
                linewidth=2, markersize=6, color=color)
    
    ax6.set_xlabel('η', fontsize=12)
    ax6.set_ylabel('<E>/N', fontsize=12)
    ax6.set_title('Mean Energy per Particle (All h)', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10, title='h (meuB)')
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/6_energy_all_h.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 6_energy_all_h.png")
    
    return results_dir


if __name__ == "__main__":
    # Set options
    eta_min = 0.1
    eta_max = 0.8
    eta_step = 0.05
    add_fine_eta_window = True  # Set to True to add fine-grained eta window (0.42-0.46)
    meuB_values = [0.0, 0.1, 0.5, 1.0]  # Include 0.0 for first three graphs, add other values as needed
    
    # Run main simulation with regular eta values (this will compile JIT)
    print("=" * 70)
    print("RUN 1: Regular eta values (JIT compilation warm-up)")
    print("=" * 70)
    all_results = main(
        eta_min=eta_min,
        eta_max=eta_max,
        eta_step=eta_step,
        add_fine_eta_window=add_fine_eta_window,
        meuB_values=meuB_values
    )
    
    # Run special calculation for eta=2, meuB=0 (only for flip ratio graph)
    # This will be faster since JIT is already compiled
    '''
    print("\n" + "=" * 70)
    print("RUN 2: Special calculation for eta=2.0, meuB=0.0 (for flip ratio graph)")
    print("=" * 70)
    eta_2_results = []
    for meuB_val in meuB_values:
        if meuB_val == 0.0:  # Only calculate for meuB=0
            print(f"\nRunning simulation for eta = 2.0, meuB = {meuB_val:.2f}")
            print("-" * 70)
            eta_2_result = run_ising_simulation(
                eta=2,
                meuB=meuB_val,
                L=L,
                K=100,
                nsweep=nsweep,
                Delta=Delta,
                kB=kB,
                max_flips=1e4,
                verbose=True
            )
            eta_2_results.append(eta_2_result)
            print()
    
    # Combine results
    all_results.extend(eta_2_results)
    '''
    # Create plots
    if all_results:
        results_dir = plot_results(all_results, L=L)
        print(f"\nAll graphs saved in: {results_dir}/")

