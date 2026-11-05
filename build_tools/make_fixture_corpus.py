"""Write the deterministic EnvTok fixture corpus to ``tests/fixtures/corpus.extxyz``.

Every structure is small, synthetic and generated from fixed parameters, so the
corpus carries no third-party licence and can be regenerated bit-for-bit.
Each entry targets a behaviour the specification requires a test for:

========================  ======================================================
name                      purpose
========================  ======================================================
nacl_prim                 ideal octahedral rocksalt
nacl_cubic                conventional cell, symmetry reuse across 8 atoms
nacl_rattled              perturbed geometry, symmetry fallback to all sites
nacl_compressed           compression bins (0.75 scaling)
nacl_stretched            stretched bond bin (1.3 scaling)
nacl_displaced            one-sided environment (asymmetry bin)
nacl_br                   mixed-ligand neighbour multiset
zns_zincblende            tetrahedral sites
zno_wurtzite              hexagonal tetrahedral sites
srtio3_cubic              octahedral B site, 12-coordinate A site
srtio3_polar              strongly off-centred Ti (extended 1.7 cutoff candidate)
cscl                      8-coordinate cube
caf2_fluorite             cube + tetrahedron in one structure
rutile_tio2               distorted octahedra, lower symmetry
fe_bcc_afm                site-dependent positive/negative initial moments
fe_bcc_ties               half-integer moments (ties-to-even rounding)
po_simple_cubic           one atom whose neighbours are all its own images
n2_in_box                 genuine one-coordinate sites
graphene_vacuum           two-dimensional sheet with vacuum
rfo_rocksalt              Z > 96 element without a tabulated covalent radius
========================  ======================================================

Labels (energy, forces, stress) and metadata are synthetic and only exist so
that export tests can check label alignment.

Usage::

    uv run python build_tools/make_fixture_corpus.py
"""

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, graphene
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.spacegroup import crystal

OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "corpus.extxyz"


def _structures() -> list[tuple[str, Atoms]]:
    nacl_prim = bulk("NaCl", "rocksalt", a=5.64)
    nacl_cubic = bulk("NaCl", "rocksalt", a=5.64, cubic=True)

    nacl_rattled = nacl_cubic.copy()
    nacl_rattled.rattle(stdev=0.08, seed=1)

    nacl_compressed = nacl_prim.copy()
    nacl_compressed.set_cell(nacl_prim.cell * 0.75, scale_atoms=True)

    nacl_stretched = nacl_prim.copy()
    nacl_stretched.set_cell(nacl_prim.cell * 1.3, scale_atoms=True)

    nacl_displaced = nacl_cubic.copy()
    positions = nacl_displaced.get_positions()
    positions[0] += (0.35, 0.0, 0.0)
    nacl_displaced.set_positions(positions)

    nacl_br = nacl_cubic.copy()
    symbols = nacl_br.get_chemical_symbols()
    symbols[symbols.index("Cl")] = "Br"
    nacl_br.set_chemical_symbols(symbols)

    srtio3 = crystal(
        ["Sr", "Ti", "O"],
        basis=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.0)],
        spacegroup=221,
        cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
    )
    srtio3_polar = srtio3.copy()
    positions = srtio3_polar.get_positions()
    titanium = srtio3_polar.get_chemical_symbols().index("Ti")
    positions[titanium] += (0.0, 0.0, 0.5)
    srtio3_polar.set_positions(positions)

    rutile = crystal(
        ["Ti", "O"],
        basis=[(0.0, 0.0, 0.0), (0.3049, 0.3049, 0.0)],
        spacegroup=136,
        cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
    )

    fe_afm = bulk("Fe", "bcc", a=2.87, cubic=True)
    fe_afm.set_initial_magnetic_moments([2.3, -2.6])
    fe_ties = bulk("Fe", "bcc", a=2.87, cubic=True)
    fe_ties.set_initial_magnetic_moments([2.5, -3.5])

    po = Atoms("Po", cell=np.eye(3) * 3.35, pbc=True)

    n2 = Atoms(
        "N2",
        positions=[(4.45, 5.0, 5.0), (5.55, 5.0, 5.0)],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )

    sheet = graphene(a=2.46, vacuum=8.0)
    sheet.pbc = True

    rfo = bulk("RfO", "rocksalt", a=4.95)

    return [
        ("nacl_prim", nacl_prim),
        ("nacl_cubic", nacl_cubic),
        ("nacl_rattled", nacl_rattled),
        ("nacl_compressed", nacl_compressed),
        ("nacl_stretched", nacl_stretched),
        ("nacl_displaced", nacl_displaced),
        ("nacl_br", nacl_br),
        ("zns_zincblende", bulk("ZnS", "zincblende", a=5.41)),
        ("zno_wurtzite", bulk("ZnO", "wurtzite", a=3.25, c=5.2)),
        ("srtio3_cubic", srtio3),
        ("srtio3_polar", srtio3_polar),
        ("cscl", bulk("CsCl", "cesiumchloride", a=4.12)),
        ("caf2_fluorite", bulk("CaF2", "fluorite", a=5.46)),
        ("rutile_tio2", rutile),
        ("fe_bcc_afm", fe_afm),
        ("fe_bcc_ties", fe_ties),
        ("po_simple_cubic", po),
        ("n2_in_box", n2),
        ("graphene_vacuum", sheet),
        ("rfo_rocksalt", rfo),
    ]


def build_corpus() -> list[Atoms]:
    """Return the fixture structures with synthetic labels and metadata."""
    rng = np.random.default_rng(20261104)
    frames = []
    for index, (name, atoms) in enumerate(_structures()):
        frame = atoms.copy()
        natoms = len(frame)
        energy = float(-3.0 * natoms + rng.normal(scale=0.1))
        forces = rng.normal(scale=0.2, size=(natoms, 3))
        stress = rng.normal(scale=0.01, size=6)
        frame.calc = SinglePointCalculator(
            frame, energy=energy, forces=forces, stress=stress
        )
        frame.info["sid"] = f"fx-{index:03d}-{name}"
        frame.info["fixture"] = name
        frame.info["split"] = "train"
        frame.info["charge"] = 0
        frame.info["method"] = "PBE"
        frames.append(frame)
    return frames


def main() -> None:
    """Write the corpus file."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames = build_corpus()
    write(OUTPUT, frames, format="extxyz")
    print(f"wrote {len(frames)} frames to {OUTPUT}")


if __name__ == "__main__":
    main()
