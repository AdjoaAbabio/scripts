
import os
import shutil
import pandas as pd

from pymatgen.core import Structure
from pymatgen.analysis.ewald import EwaldSummation


# Folder containing the ordered CIF structures

INPUT_DIR = "Na0.886V2O5_n1_1x1x1_integer_configs"

# Folder where Ewald results and selected structures are saved
OUTPUT_DIR = "ewald_screened_N0.886V2O5"

# Number of lowest-Ewald structures to keep
N_BEST = 3


FIXED_CHARGES = {
    "Na": 1.0,
    "O": -2.0,
}

# ------------------------------------------------------------
# Element whose charge will be calculated automatically
#
# Example:
#     "V"  for Na-V-O
#     "Mn" for Na-Mn-O
#     "Co" for Na-Co-O
# ------------------------------------------------------------

VARIABLE_CHARGE_ELEMENT = "V"


# ============================================================
# FUNCTION: CALCULATE VARIABLE CHARGE
# ============================================================

def calculate_variable_charge(structure):
    """
    Calculate the oxidation state of the variable-charge
    element using charge neutrality.

    The total charge of the periodic structure must be zero:

        sum(number_of_atoms * oxidation_state) = 0

    Therefore:

        q_variable =
        - sum(fixed_atoms * fixed_charge)
          / number_of_variable_atoms
    """

    composition = structure.composition

    variable_element = VARIABLE_CHARGE_ELEMENT

    if variable_element not in composition:
        raise ValueError(
            f"{variable_element} was not found in the structure."
        )

    variable_amount = composition[variable_element]

    total_fixed_charge = 0.0

    for element, charge in FIXED_CHARGES.items():

        if element in composition:

            amount = composition[element]

            total_fixed_charge += amount * charge

    variable_charge = -total_fixed_charge / variable_amount

    return variable_charge


# ============================================================
# FUNCTION: ASSIGN OXIDATION STATES
# ============================================================

def assign_charges(structure):
    """
    Assign fixed oxidation states and automatically calculate
    the oxidation state of the variable-charge element.
    """

    charged_structure = structure.copy()

    # Calculate variable element charge
    variable_charge = calculate_variable_charge(
        charged_structure
    )

    # Build complete oxidation-state dictionary
    oxidation_states = dict(FIXED_CHARGES)

    oxidation_states[
        VARIABLE_CHARGE_ELEMENT
    ] = variable_charge

    # Check that every element in the structure has a charge
    for element in charged_structure.composition.elements:

        symbol = element.symbol

        if symbol not in oxidation_states:

            raise ValueError(
                f"No oxidation state defined for element: "
                f"{symbol}"
            )

    # Assign oxidation states using pymatgen
    charged_structure.add_oxidation_state_by_element(
        oxidation_states
    )

    return charged_structure, variable_charge


# ============================================================
# FUNCTION: CALCULATE EWALD ENERGY
# ============================================================

def calculate_ewald(structure):
    """
    Calculate the total Ewald energy and energy per formula unit.
    """

    charged_structure, variable_charge = assign_charges(
        structure
    )

    # Pymatgen Ewald summation
    ewald = EwaldSummation(
        charged_structure
    )

    total_energy = ewald.total_energy

    # --------------------------------------------------------
    # Determine number of reduced formula units
    # --------------------------------------------------------

    reduced_composition, factor = (
        structure.composition
        .get_reduced_composition_and_factor()
    )

    energy_per_formula = (
        total_energy / factor
    )

    return (
        total_energy,
        energy_per_formula,
        variable_charge,
    )


# ============================================================
# FUNCTION: SCREEN ALL STRUCTURES
# ============================================================

def screen_structures():

    # --------------------------------------------------------
    # Check input directory
    # --------------------------------------------------------

    if not os.path.isdir(INPUT_DIR):

        raise FileNotFoundError(
            f"\nInput directory does not exist:\n"
            f"{INPUT_DIR}\n"
        )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    best_dir = os.path.join(
        OUTPUT_DIR,
        "best_structures"
    )

    os.makedirs(
        best_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find CIF files
    # --------------------------------------------------------

    cif_files = sorted(
        filename
        for filename in os.listdir(INPUT_DIR)
        if filename.lower().endswith(".cif")
    )

    print()
    print("=" * 70)
    print("GENERALIZED EWALD SCREENING")
    print("=" * 70)

    print(
        f"Input directory          : {INPUT_DIR}"
    )

    print(
        f"Number of CIF structures  : {len(cif_files)}"
    )

    print(
        f"Variable-charge element   : "
        f"{VARIABLE_CHARGE_ELEMENT}"
    )

    print(
        f"Number of structures kept: {N_BEST}"
    )

    print()

    if not cif_files:

        print("No CIF files were found.")

        return

    # ========================================================
    # CALCULATE EWALD ENERGY
    # ========================================================

    results = []

    for index, filename in enumerate(
        cif_files,
        start=1
    ):

        filepath = os.path.join(
            INPUT_DIR,
            filename
        )

        print(
            f"[{index}/{len(cif_files)}] "
            f"{filename}"
        )

        try:

            # ------------------------------------------------
            # Read structure
            # ------------------------------------------------

            structure = Structure.from_file(
                filepath
            )

            # ------------------------------------------------
            # Calculate Ewald energy
            # ------------------------------------------------

            (
                total_energy,
                energy_per_formula,
                variable_charge,
            ) = calculate_ewald(
                structure
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            results.append({

                "filename":
                    filename,

                "variable_element":
                    VARIABLE_CHARGE_ELEMENT,

                "variable_charge":
                    variable_charge,

                "ewald_energy_eV":
                    total_energy,

                "ewald_energy_per_formula_eV":
                    energy_per_formula,
            })

            print(
                f"    {VARIABLE_CHARGE_ELEMENT} charge = "
                f"{variable_charge:.6f}"
            )

            print(
                f"    Ewald energy = "
                f"{total_energy:.6f} eV"
            )

            print(
                f"    Ewald/formula = "
                f"{energy_per_formula:.6f} eV"
            )

        except Exception as error:

            print(
                f"    ERROR: {type(error).__name__}"
            )

            print(
                f"    {error}"
            )

            results.append({

                "filename":
                    filename,

                "variable_element":
                    VARIABLE_CHARGE_ELEMENT,

                "variable_charge":
                    None,

                "ewald_energy_eV":
                    None,

                "ewald_energy_per_formula_eV":
                    None,
            })

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(results)

    # Keep only successful calculations
    valid_df = df.dropna(
        subset=[
            "ewald_energy_per_formula_eV"
        ]
    ).copy()

    # ========================================================
    # SORT BY EWALD ENERGY
    # ========================================================

    valid_df.sort_values(
        by="ewald_energy_per_formula_eV",
        ascending=True,
        inplace=True
    )

    # ========================================================
    # ADD RANK
    # ========================================================

    valid_df.insert(
        0,
        "rank",
        range(
            1,
            len(valid_df) + 1
        )
    )

    # ========================================================
    # SAVE COMPLETE RANKING
    # ========================================================

    ranking_file = os.path.join(
        OUTPUT_DIR,
        "ewald_ranking.csv"
    )

    valid_df.to_csv(
        ranking_file,
        index=False
    )

    # ========================================================
    # SELECT LOWEST N_BEST
    # ========================================================

    best_df = valid_df.head(
        N_BEST
    ).copy()

    # ========================================================
    # COPY BEST CIF FILES
    # ========================================================

    for _, row in best_df.iterrows():

        source = os.path.join(
            INPUT_DIR,
            row["filename"]
        )

        destination = os.path.join(
            best_dir,
            row["filename"]
        )

        shutil.copy2(
            source,
            destination
        )

    # ========================================================
    # SAVE BEST STRUCTURES CSV
    # ========================================================

    best_csv = os.path.join(
        OUTPUT_DIR,
        "ewald_best_structures.csv"
    )

    best_df.to_csv(
        best_csv,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EWALD SCREENING COMPLETE")
    print("=" * 70)

    print(
        f"Successful calculations : "
        f"{len(valid_df)}"
    )

    print(
        f"Structures selected      : "
        f"{len(best_df)}"
    )

    print()
    print(
        f"Complete ranking:"
    )

    print(
        f"    {ranking_file}"
    )

    print()
    print(
        f"Best structures:"
    )

    print(
        f"    {best_dir}"
    )

    print()
    print(
        f"Best structures CSV:"
    )

    print(
        f"    {best_csv}"
    )

    # ========================================================
    # DISPLAY TOP 10
    # ========================================================

    print()
    print("=" * 70)
    print("TOP 10 LOWEST-EWALD STRUCTURES")
    print("=" * 70)

    print()

    print(
        best_df.head(10).to_string(
            index=False
        )
    )

    print()


# ------------------------------------------------------------------------
# RUN SCRIPT
# -----------------------------------------------------------------------

if __name__ == "__main__":

    screen_structures()









