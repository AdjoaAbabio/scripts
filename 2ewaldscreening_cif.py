from pathlib import Path
import shutil
import csv

import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.analysis.ewald import EwaldSummation


INPUT_DIR = Path("Na0.886V2O5_n1_1x1x1_integer_configs")

OUTPUT_DIR = Path("ewald_screened_N0.886V2O5")

# Number of structures to keep
N_BEST = 3

# Fixed oxidation states
FIXED_CHARGES = {
    "Na": 1.0,
    "O": -2.0,
}

# Element whose oxidation state is calculated
# automatically from charge neutrality
VARIABLE_CHARGE_ELEMENT = "V"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_DIR = OUTPUT_DIR / "best_structures"
BEST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND CIF FILES
# ============================================================

cif_files = sorted(INPUT_DIR.glob("*.cif"))

if not cif_files:
    raise FileNotFoundError(
        f"No CIF files found in: {INPUT_DIR}"
    )


print("=" * 70)
print("EWALD ENERGY SCREENING")
print("=" * 70)
print(f"Input directory           : {INPUT_DIR}")
print(f"Output directory          : {OUTPUT_DIR}")
print(f"Number of CIF structures  : {len(cif_files)}")
print(f"Variable-charge element   : {VARIABLE_CHARGE_ELEMENT}")
print(f"Number of structures kept: {min(N_BEST, len(cif_files))}")
print("=" * 70)


# ============================================================
# FUNCTION:
# CALCULATE VARIABLE OXIDATION STATE
# ============================================================

def calculate_variable_charge(structure):
    """
    Calculate the oxidation state of the variable element
    from charge neutrality.

    Fixed:
        Na = +1
        O  = -2

    Variable:
        V = calculated automatically
    """

    composition = structure.composition

    n_variable = composition[VARIABLE_CHARGE_ELEMENT]

    if n_variable == 0:
        raise ValueError(
            f"{VARIABLE_CHARGE_ELEMENT} not found in structure."
        )

    fixed_charge_total = 0.0

    for element, charge in FIXED_CHARGES.items():

        n_atoms = composition[element]

        fixed_charge_total += n_atoms * charge

    variable_charge = -fixed_charge_total / n_variable

    return variable_charge


# ============================================================
# CALCULATE EWALD ENERGIES
# ============================================================

results = []

for structure_index, cif_file in enumerate(cif_files, start=1):

    print(
        f"[{structure_index}/{len(cif_files)}] "
        f"Processing {cif_file.name}"
    )

    try:

        # ----------------------------------------------------
        # Read CIF
        # ----------------------------------------------------

        structure = Structure.from_file(cif_file)

        # ----------------------------------------------------
        # Calculate variable oxidation state
        # ----------------------------------------------------

        variable_charge = calculate_variable_charge(
            structure
        )

        # ----------------------------------------------------
        # Assign oxidation states
        # ----------------------------------------------------

        oxidation_states = dict(FIXED_CHARGES)

        oxidation_states[
            VARIABLE_CHARGE_ELEMENT
        ] = variable_charge

        charged_structure = structure.copy()

        charged_structure.add_oxidation_state_by_element(
            oxidation_states
        )

        # ----------------------------------------------------
        # Calculate TOTAL EWALD ENERGY
        # ----------------------------------------------------

        ewald = EwaldSummation(
            charged_structure
        )

        total_ewald_energy = ewald.total_energy

        # ----------------------------------------------------
        # Calculate number of formula units
        # ----------------------------------------------------

        reduced_composition, factor = (
            structure.composition
            .get_reduced_composition_and_factor()
        )

        ewald_per_formula = (
            total_ewald_energy / factor
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append(
            {
                "structure_index": structure_index,
                "filename": cif_file.name,
                "variable_element": VARIABLE_CHARGE_ELEMENT,
                "variable_charge": variable_charge,

                # TOTAL Ewald energy
                "ewald_energy_eV": total_ewald_energy,

                # Kept only as additional information
                "ewald_energy_per_formula_eV":
                    ewald_per_formula,
            }
        )

    except Exception as error:

        print(
            f"    ERROR processing "
            f"{cif_file.name}: {error}"
        )


# ============================================================
# SORT BY TOTAL EWALD ENERGY
#
# LOWEST TOTAL EWALD ENERGY = RANK 1
# ============================================================

results.sort(
    key=lambda x: x["ewald_energy_eV"]
)


# ============================================================
# ASSIGN RANKS
# ============================================================

for rank, result in enumerate(
    results,
    start=1
):

    result["rank"] = rank


# ============================================================
# SAVE COMPLETE RANKING CSV
# ============================================================

ranking_csv = (
    OUTPUT_DIR /
    "ewald_energy_ranking.csv"
)

fieldnames = [
    "rank",
    "structure_index",
    "filename",
    "variable_element",
    "variable_charge",
    "ewald_energy_eV",
    "ewald_energy_per_formula_eV",
]


with open(
    ranking_csv,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for result in results:

        writer.writerow(result)


# ============================================================
# SELECT LOWEST-ENERGY STRUCTURES
# ============================================================

best_results = results[:N_BEST]


# ============================================================
# COPY BEST CIF FILES
# ============================================================

for result in best_results:

    source = (
        INPUT_DIR /
        result["filename"]
    )

    destination = (
        BEST_DIR /
        result["filename"]
    )

    shutil.copy2(
        source,
        destination
    )


# ============================================================
# SAVE BEST STRUCTURES CSV
# ============================================================

best_csv = (
    OUTPUT_DIR /
    "best_structures.csv"
)


with open(
    best_csv,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for result in best_results:

        writer.writerow(result)


# ============================================================
# PLOT:
# RANK vs TOTAL EWALD ENERGY
# ============================================================

# IMPORTANT:
# X-axis = RANK
# Y-axis = TOTAL EWALD ENERGY
#
# Rank 1 = lowest Ewald energy
# Highest rank = highest Ewald energy

ranks = [
    result["rank"]
    for result in results
]

energies = [
    result["ewald_energy_eV"]
    for result in results
]


# Energies of the selected structures

best_ranks = [
    result["rank"]
    for result in best_results
]

best_energies = [
    result["ewald_energy_eV"]
    for result in best_results
]


# ------------------------------------------------------------
# CREATE PLOT
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))


# Plot ALL structures

plt.scatter(
    ranks,
    energies,
    label="All structures"
)


# Plot SELECTED structures

plt.scatter(
    best_ranks,
    best_energies,
    s=80,
    label=f"Selected lowest {len(best_results)}"
)


# ------------------------------------------------------------
# AXIS LABELS
# ------------------------------------------------------------

plt.xlabel(
    "Ewald energy rank"
)

plt.ylabel(
    "Ewald energy (eV)"
)


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

plt.title(
    "Ewald Energy Ranking of Ordered Structures"
)


# ------------------------------------------------------------
# GRID
# ------------------------------------------------------------

plt.grid(
    True,
    alpha=0.3
)


# ------------------------------------------------------------
# LEGEND
# ------------------------------------------------------------

plt.legend()


# ------------------------------------------------------------
# SAVE FIGURE
# ------------------------------------------------------------

plt.tight_layout()

plot_file = (
    OUTPUT_DIR /
    "ewald_energy_vs_rank.png"
)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("EWALD SCREENING COMPLETE")
print("=" * 70)

print(
    f"Structures successfully processed : "
    f"{len(results)}"
)

print(
    f"Lowest Ewald energy               : "
    f"{results[0]['ewald_energy_eV']:.6f} eV"
)

print(
    f"Highest Ewald energy              : "
    f"{results[-1]['ewald_energy_eV']:.6f} eV"
)

print()
print("Selected structures:")
print("-" * 70)


for result in best_results:

    print(
        f"Rank {result['rank']:>4} | "
        f"{result['filename']} | "
        f"{result['ewald_energy_eV']:.6f} eV"
    )


print()
print(f"Ranking CSV : {ranking_csv}")
print(f"Best CSV    : {best_csv}")
print(f"Best CIFs   : {BEST_DIR}")
print(f"Plot        : {plot_file}")

print("=" * 70)
