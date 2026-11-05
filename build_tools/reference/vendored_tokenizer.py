# Vendored from em819/omat24-chemenv-subset, file
# 01_token_construction/tokenize_omat24.py at commit
# 723696de2c26584b8a907ef900690c0caf3c7b68.
#
# MIT License, Copyright (c) 2026 Evgeny Moerman. The full licence text is in
# LICENSES/omat24-chemenv-subset-MIT.txt at the root of this repository.
#
# Lines 198-517 of the upstream file (the per-site descriptor functions and
# their constants) are copied VERBATIM below. Only the MPI/HDF5/input-backend
# plumbing is omitted. `tokenize_atoms` at the end reproduces the body of the
# upstream process_single_structure() (lines 529-659) with the MPI rank, the
# HDF5 write and the dataset lookup removed; every scientific statement is
# unchanged. This file is used ONLY to generate golden fixtures in the pinned
# reference environment (see README.md) and is not part of the envtok package.
from pymatgen.core import Structure
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import Counter

from ase.data import covalent_radii, atomic_numbers
import numpy as np

# ---------------------------------------------------------------- verbatim ---
lgf = None

def get_complete_coordination_info(lse, structure, site_index, strategy=None):
    neighbors_info = None
    if lse is not None:
        # Get coordination environment
        coord_env = lse.coordination_environments[site_index]
        neighbors_info = lse.neighbors_sets[site_index]

    if lse is not None and neighbors_info:           # truthy: handles both None and []
        result = {
            'central_atom': structure[site_index].specie.symbol,
            'coordination_environments': coord_env,
            'neighbor_elements': [n.specie.symbol for n in neighbors_info[0].neighb_sites],
            #The neighbour objects THEMSELVES, so that the distance-ratio fields are
            #computed from exactly the shell that produced 'neighbor_elements'.
            'neighbor_sites': list(neighbors_info[0].neighb_sites),
        }
    else:
        result = {
            'central_atom': structure[site_index].specie.symbol,
            'coordination_environments': 'INVALID',
            'neighbor_elements': 'INVALID',
            'neighbor_sites': []
            }

    return result


# ----------------------------------------------------------------------------
# Adaptive per-atom distance cutoff (two-cutoff selector)
#
# chemenv's SimplestChemenvStrategy selects neighbours with a distance cutoff
# that is RELATIVE to the nearest neighbour. The default (1.4) therefore
# under-coordinates atoms sitting in shells of very different bond lengths
# (e.g. a phosphate O bonded to a short P and a long Ca, or N2 trapped in a
# metal cage): once distances are normalised to the closest contact, the longer
# but genuine bonds fall past the cutoff and the atom collapses to S:1.
#
# We borrow VoronoiNN's solid-angle weight purely as a DISCOVERY signal to
# decide, per atom, how far chemenv should reach — while leaving chemenv as the
# geometry CLASSIFIER. For each representative site we build the neighbour set at
# the default cutoff (1.4) and at an extended cutoff (1.7) and adopt the extended
# set only if at least one neighbour it ADDS has a normalised Voronoi solid-angle
# weight >= VORONOI_PROMINENCE_THRESHOLD (i.e. that extra contact occupies a real
# fraction of the central atom's surroundings, not a screened sliver). Otherwise
# we keep the default 1.4 set. The choice is bounded to [1.4, 1.7]: it can never
# collapse below 1.4 (avoids CrystalNN-style under-coordination of rattled
# structures) nor extend past 1.7 (avoids running into the continuum of a dense
# metal). The selection is per ATOM and based only on composition/geometry, so it
# is applied identically to every dataset (MatPES, OMat24) with no data-origin
# branching.
#
# These four constants also appear in stage 10.2's analyze_predictions.py, which
# has to produce the same tokens. Keep the two copies in sync.
DISTANCE_CUTOFF_DEFAULT = 1.4
DISTANCE_CUTOFF_EXTENDED = 1.7
ANGLE_CUTOFF = 0.3
VORONOI_PROMINENCE_THRESHOLD = 0.45


# ----------------------------------------------------------------------------
# Distance-ratio field (rmin) and the asymmetry field A.
#
# The rest of the token is scale-invariant: chemenv rescales the coordination
# polyhedron before scoring it, so a structure and the same structure compressed
# by 20% give the same token. These two fields put the length scale back.
#
# For every contact i of the neighbour set we form a reduced length
#
#     r_i = d_i / (R_cov[central] + R_cov[neighbour_i])
#
# using Cordero covalent radii, so r is about 1 at a relaxed bond, below 1 when
# compressed and above 1 when stretched. The smallest r over the shell (rmin) is
# binned by DIST_RATIO_MIN_EDGES into five states: two compression bands, a
# near-equilibrium band, and a stretched band.
#
# Sites with no usable neighbour set, or whose neighbours have no tabulated
# covalent radius, get DIST_BIN_UNDEFINED so they all collapse into one class
# instead of fragmenting. 999 cannot collide with a real bin index.
#
# DIST_RATIO_MAX_EDGES is kept so the stored rmax stays re-binnable, but rmax is
# not part of the token: chemenv picks the neighbour shell with a cutoff relative
# to the nearest contact, so a large rmax means a wide shell rather than a
# stretched bond.
DIST_RATIO_MIN_EDGES = (0.60, 0.70, 0.85, 1.15)
DIST_RATIO_MAX_EDGES = (0.85, 1.15, 1.50)
DIST_BIN_UNDEFINED = 999

# ---------------------------------------------------------------------------
# The asymmetry field A.
#
#     u_ij = d_ij / (R_cov[i] + R_cov[j])      the same reduced length as above
#     w(u) = exp(-(u - 1) / A_LAMBDA)          1 at contact, larger as u falls
#     A_i  = | sum_j w(u_ij) * rhat_ij |       magnitude of the vector sum
#
# A measures how one-sided a site's contacts are. Force is a vector sum of pair
# terms, but every other field in the token is a scalar or an unordered set: a
# symmetric compressed shell and the same contacts pushed to one side look
# identical to them, yet only the second carries a net force. A separates them.
#
# A_LAMBDA sets how sharply the weighting favours short contacts. Small values
# make A collapse onto the single shortest contact, i.e. a finer rmin; large
# values wash the compression weighting out.
#
# A cannot overflow. u is always >= 0 (contacts with d <= 0 are skipped), so
#     w(u) < exp(1 / A_LAMBDA) = e^10
# and A stays below n_neighbours * e^10. np.exp here can never return inf.
A_LAMBDA = 0.10
A_EDGES = (1.0, 3.0)

#Highest Z for which Cordero et al. (2008) actually report a covalent radius.
#ase.data.covalent_radii is indexed all the way to Z = 118, but Z = 97..118 are
#PADDED with a 2.0 A placeholder rather than left as NaN, so a bound of
#len(covalent_radii) alone would accept a fabricated radius. See _covalent_radius().
CORDERO_MAX_Z = 96


def _covalent_radius(symbol):
    """Cordero covalent radius in Angstrom, or None if the element is not tabulated.

    The Z <= CORDERO_MAX_Z bound is what actually makes the "not tabulated" branch
    reachable: ASE pads Z = 97..118 with a 2.0 A placeholder instead of NaN, so
    without it a transactinide would be silently binned against a fabricated
    reference distance rather than reported undefined. This is a no-op for
    OMat24/MatPES -- nothing above Cm (96) occurs in either, and Bk..Lr (97..103)
    are dropped upstream by contains_rare_earths() anyway -- so no existing site's
    bins change. It only keeps DIST_BIN_UNDEFINED honest on future data.
    """
    z = atomic_numbers.get(symbol)
    if z is None or z <= 0 or z > CORDERO_MAX_Z or z >= len(covalent_radii):
        return None
    r = float(covalent_radii[z])
    if not np.isfinite(r) or r <= 0.0:
        return None
    return r


def _distance_ratio_bins(structure, site_index, neighbor_sites):
    """Return (ratio_min, ratio_max, bin_min, bin_max, a_raw, a_bin) for one site.

    _and_A: the asymmetry A is accumulated INSIDE the same loop, over the same
    shell, under the same skip rules, so A can never describe a different set of
    contacts from the ones that produced ratio_min/ratio_max. That is also why the
    CrystalNN rescue path gets A for free -- it reuses this helper unchanged.

    `neighbor_sites` MUST be the same neighbour objects that produced
    coord_elements, so that the distance fields and the composition field can
    never describe two different shells. Each entry only needs `.coords`
    (cartesian and image-resolved, i.e. already unwrapped across the periodic
    boundary) and `.specie.symbol` -- true both for chemenv's neighb_sites and
    for the sites returned by CrystalNN.get_nn_info, which is how the rescue path
    below can reuse this helper unchanged.

    Distances are measured the same way as in _neighbor_keyset(): the norm of the
    neighbour image coordinate minus the central site coordinate. Contacts whose
    element has no tabulated radius are skipped; if that leaves no usable contact
    the site is reported undefined (NaN ratios, 999 bins) instead of being
    silently binned from a partial shell.
    """
    r_central = _covalent_radius(structure[site_index].specie.symbol)
    ratios = []
    #Weighted sum of UNIT vectors from the central site to each usable contact.
    #Its magnitude is A; it stays exactly zero for a centrosymmetric shell.
    a_vec = np.zeros(3, dtype=float)
    if r_central is not None:
        center = np.array(structure[site_index].coords)
        for nb in neighbor_sites:
            r_nb = _covalent_radius(nb.specie.symbol)
            if r_nb is None:
                continue
            d0 = r_central + r_nb
            #vec hoisted out of the norm: A needs the direction as well as the length
            vec = np.array(nb.coords) - center
            dist = float(np.linalg.norm(vec))
            if not np.isfinite(dist) or dist <= 0.0:
                #Guard against a degenerate self-contact / duplicated site.
                continue
            ratio = dist / d0
            ratios.append(ratio)
            a_vec = a_vec + np.exp(-(ratio - 1.0) / A_LAMBDA) * (vec / dist)
    if not ratios:
        #A is undefined for exactly the sites whose ratios are, and shares the
        #999 sentinel so they all collapse into ONE class rather than fragmenting.
        return (float(np.nan), float(np.nan), DIST_BIN_UNDEFINED, DIST_BIN_UNDEFINED,
                float(np.nan), DIST_BIN_UNDEFINED)
    ratio_min = min(ratios)
    ratio_max = max(ratios)
    a_raw = float(np.linalg.norm(a_vec))
    return (float(ratio_min), float(ratio_max),
            int(np.digitize(ratio_min, DIST_RATIO_MIN_EDGES)),
            int(np.digitize(ratio_max, DIST_RATIO_MAX_EDGES)),
            a_raw, int(np.digitize(a_raw, A_EDGES)))


def _neighbor_keyset(structure, site_index, neighbors_set):
    """Multiset of (element, distance-rounded-to-0.01-A) for a chemenv neighbour set."""
    center = np.array(structure[site_index].coords)
    keys = Counter()
    for nb in neighbors_set.neighb_sites:
        dist = round(float(np.linalg.norm(np.array(nb.coords) - center)), 2)
        keys[(nb.specie.symbol, dist)] += 1
    return keys


def select_site_lse(lse_default, lse_extended, structure, site_index, vnn,
                    threshold=VORONOI_PROMINENCE_THRESHOLD):
    """Pick the 1.4 or 1.7 LightStructureEnvironments for one site (see block header).

    Returns lse_extended iff the extended neighbour set adds at least one contact
    whose normalised VoronoiNN solid-angle weight is >= `threshold`; otherwise
    returns lse_default. Falls back to lse_default on any failure (missing
    extended set, Voronoi construction error, etc.) so the selector can never make
    a site worse than the default-cutoff behaviour of the original script.
    """
    if lse_default is None or lse_extended is None:
        return lse_default
    try:
        ns_default = lse_default.neighbors_sets[site_index]
        ns_extended = lse_extended.neighbors_sets[site_index]
    except Exception:
        return lse_default
    if not ns_extended:
        return lse_default
    k_default = _neighbor_keyset(structure, site_index, ns_default[0]) if ns_default else Counter()
    k_extended = _neighbor_keyset(structure, site_index, ns_extended[0])
    #Neighbours present in the 1.7 set beyond the 1.4 set.
    added = [k for k in (set(k_extended) | set(k_default))
             if k_extended.get(k, 0) > k_default.get(k, 0)]
    if not added:
        return lse_default
    try:
        center = np.array(structure[site_index].coords)
        weight_map = {}
        for n in vnn.get_nn_info(structure, site_index):
            dist = round(float(np.linalg.norm(np.array(n['site'].coords) - center)), 2)
            key = (n['site'].specie.symbol, dist)
            if n['weight'] > weight_map.get(key, 0.0):
                weight_map[key] = n['weight']
    except Exception:
        return lse_default
    prominence = max((weight_map.get(k, 0.0) for k in added), default=0.0)
    return lse_extended if prominence >= threshold else lse_default


def build_coord_descriptor(lse, structure, site_index, cnn):
    """Return (coord_elements, coord_symbol, sym_measure, perturb_scale,
    ratio_min, ratio_max, dist_bin_min, dist_bin_max, a_raw, a_bin) for a site.
    Note dist_bin_max is still COMPUTED and returned (and stored as an HDF5 column);
    it is simply no longer part of the env_id -- a_bin took its slot.

    This is the single source of truth for the coordination part of the env_id.
    It returns 10 values: the four coordination fields, the raw min/max distance
    ratios and their bins, and the raw asymmetry and its bin. Stage 10.2 has the
    same function and must return the same values, or tokens from the two sides
    stop being comparable.

    The distance ratios are computed from coord_info['neighbor_sites'], i.e. the
    very shell whose elements form coord_elements, so composition and distance
    always refer to the same neighbour set. The CrystalNN rescue path below does
    the same with its own neighbour list.

    Primary source is the chemenv coordination environment (coord_symbol like
    'O:6', distortion-bucketed perturb_scale from the continuous symmetry measure).
    When chemenv assigns nothing for the site — no environment, or the whole
    structure's Voronoi construction failed (common for 2D / vacuum-containing
    cells, where chemenv raises and lse is None) — we fall back to CrystalNN so the
    neighbour composition and coordination number are still recorded instead of
    being thrown away. Such rescued sites are labelled with chemenv's OWN
    'UNKNOWN:<cn>' symbol (the same token chemenv emits when it finds a neighbour
    set but matches no ideal geometry) with perturb_scale = 999, so that every
    'CN n, no geometry' atom falls into ONE class for classification regardless of
    whether chemenv or CrystalNN supplied the neighbours. Only if CrystalNN also
    fails is the descriptor fully INVALID.
    """
    coord_info = get_complete_coordination_info(lse, structure, site_index)
    coord_env = coord_info['coordination_environments']
    if coord_env != 'INVALID' and coord_env is not None and len(coord_env) > 0:
        coord_symbol = coord_env[0]['ce_symbol']
        sym_measure = coord_env[0]['csm']
        coord_elements = "-".join(sorted(list(coord_info['neighbor_elements'])))
        if sym_measure is not None:
            if sym_measure <= 1.0:
                perturb_scale = 0
            elif sym_measure <= 5:
                perturb_scale = 1
            elif sym_measure < 10:
                perturb_scale = 2
            else:
                perturb_scale = 3
        else:
            perturb_scale = 999
            sym_measure = float(np.nan)
        ratio_min, ratio_max, bin_min, bin_max, a_raw, a_bin = _distance_ratio_bins(
            structure, site_index, coord_info['neighbor_sites'])
        return (coord_elements, coord_symbol, float(sym_measure), perturb_scale,
                ratio_min, ratio_max, bin_min, bin_max, a_raw, a_bin)

    #Rescue: chemenv could not assign an environment. Recover the neighbour
    #composition + coordination number with CrystalNN (robust to the low-dimensional
    #/ pathological-Voronoi cases that defeat chemenv).
    try:
        nn_info = cnn.get_nn_info(structure, site_index)
        neigh = sorted(n['site'].specie.symbol for n in nn_info)
        if len(neigh) > 0:
            ratio_min, ratio_max, bin_min, bin_max, a_raw, a_bin = _distance_ratio_bins(
                structure, site_index, [n['site'] for n in nn_info])
            return ("-".join(neigh), f"UNKNOWN:{len(neigh)}", float(np.nan), 999,
                    ratio_min, ratio_max, bin_min, bin_max, a_raw, a_bin)
    except Exception:
        pass

    return ('INVALID', 'INVALID', float(np.nan), 999,
            float(np.nan), float(np.nan), DIST_BIN_UNDEFINED, DIST_BIN_UNDEFINED,
            float(np.nan), DIST_BIN_UNDEFINED)

def contains_rare_earths(structure):
    rare_earth_elements = {'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr'} 
    if bool(set(structure.get_chemical_symbols()).intersection(rare_earth_elements)):
        return True
    else:
        return False
# ------------------------------------------------------------ end verbatim ---


def make_local_geometry_finder():
    """Upstream main(), lines 840-852: the LocalGeometryFinder setup."""
    lgf = LocalGeometryFinder()
    lgf.setup_parameters(centering_type='centroid', include_central_site_in_centroid=True,
                         structure_refinement=LocalGeometryFinder.STRUCTURE_REFINEMENT_NONE)
    return lgf


def tokenize_atoms(atoms, lgf, mat_id):
    """Upstream process_single_structure() lines 529-659, without MPI/HDF5.

    Returns the list of per-atom row dicts the upstream function would write,
    or None where upstream skips the structure (rare earths). The omat_subset
    columns are omitted because they only label the input shard.
    """
    env_list = []
    atoms_formula = atoms.get_chemical_formula()
    if contains_rare_earths(atoms):
        return None

    magmoms = atoms.get_initial_magnetic_moments()

    pymatgen_atoms = Structure.from_ase_atoms(atoms)
    pymatgen_atoms.add_site_property("magmom", magmoms)

    n_sites = len(pymatgen_atoms)
    symmetry_fallback = False
    try:
        sym_dataset = SpacegroupAnalyzer(pymatgen_atoms).get_symmetry_dataset()
        if sym_dataset is None:
            raise ValueError("spglib returned no symmetry dataset")
        equivalent_atoms = [int(x) for x in sym_dataset.equivalent_atoms]
        if len(equivalent_atoms) != n_sites:
            raise ValueError(f"equivalent_atoms length {len(equivalent_atoms)} != {n_sites} sites")
    except Exception as e:
        symmetry_fallback = True
        equivalent_atoms = list(range(n_sites))
    unique_reps = list(dict.fromkeys(equivalent_atoms))

    lgf.setup_structure(pymatgen_atoms)
    chemenv_failed = False
    try:
        se = lgf.compute_structure_environments(only_cations=False, only_indices=unique_reps)
        lse_default = LightStructureEnvironments.from_structure_environments(
            strategy=SimplestChemenvStrategy(distance_cutoff=DISTANCE_CUTOFF_DEFAULT, angle_cutoff=ANGLE_CUTOFF),
            structure_environments=se)
        lse_extended = LightStructureEnvironments.from_structure_environments(
            strategy=SimplestChemenvStrategy(distance_cutoff=DISTANCE_CUTOFF_EXTENDED, angle_cutoff=ANGLE_CUTOFF),
            structure_environments=se)
    except Exception as e:
        chemenv_failed = True
        lse_default = None
        lse_extended = None

    cnn = CrystalNN()
    vnn = VoronoiNN(tol=0.0)
    rep_coord_desc = {}
    rep_chosen = {}
    for rep in unique_reps:
        chosen_lse = select_site_lse(lse_default, lse_extended, pymatgen_atoms, rep, vnn)
        rep_chosen[rep] = ('none' if chosen_lse is None else
                           'extended' if chosen_lse is lse_extended and chosen_lse is not lse_default
                           else 'default')
        rep_coord_desc[rep] = build_coord_descriptor(chosen_lse, pymatgen_atoms, rep, cnn)

    for site_index in range(len(pymatgen_atoms)):
        central_element = pymatgen_atoms[site_index].specie.symbol
        spin = round(np.abs(magmoms[site_index]))
        (coord_elements, coord_symbol, sym_measure, perturb_scale,
         dist_ratio_min, dist_ratio_max, dist_bin_min, dist_bin_max,
         a_raw, a_bin) = rep_coord_desc[equivalent_atoms[site_index]]
        env_id = (central_element + '_' + str(spin) + '_' + coord_elements + '_' + coord_symbol
                  + '_' + str(perturb_scale) + '_' + str(dist_bin_min) + '_' + str(a_bin))
        env_list.append(
                {'env_id': env_id,
                 'central_element' : str(central_element),
                 'spin' : int(spin),
                 'coord_symbol' : str(coord_symbol),
                 'coord_elements' : str(coord_elements),
                 'sym_measure' : float(sym_measure),
                 'dist_ratio_min' : float(dist_ratio_min),
                 'dist_ratio_max' : float(dist_ratio_max),
                 'a_raw' : float(a_raw),
                 'a_bin' : int(a_bin),
                 'dist_bin_max' : int(dist_bin_max),
                 'isite' : int(site_index),
                 'mat_id' : str(mat_id),
                 # Diagnostics added for fixture comparison (not upstream columns):
                 'rep' : int(equivalent_atoms[site_index]),
                 'chosen_cutoff' : rep_chosen[equivalent_atoms[site_index]],
                 'symmetry_fallback' : symmetry_fallback,
                 'chemenv_failed' : chemenv_failed,
                 }
                )
    return env_list
