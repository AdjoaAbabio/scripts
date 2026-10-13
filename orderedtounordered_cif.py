#!/usr/bin/env python3

"""
ORDERED CIF -> INTEGER Na CONFIGURATIONS

For a CIF containing partially occupied Na sites such as:

    Na0.64
    Na0.31

the script interprets:

    Na1 -> experimental occupation = 64%
    Na2 -> experimental occupation = 31%

The script:

1. Reads the ordered/partially occupied CIF.
2. Identifies Na1 and Na2 from their occupations.
3. Counts the number of Na1 and Na2 sites.
4. Tests supercell multipliers n = 1 ... maximum_n.
5. Calculates the closest integer occupation to:
       Na1 = 64%
       Na2 = 31%
6. Calculates the number of possible arrangements:
       C(N, k)
7. Calculates:
       Na1 arrangements x Na2 arrangements
8. Lets the user select n.
9. Lets the user select the supercell direction.
10. Generates every possible explicit Na configuration.
11. Writes every configuration as an individual CIF.

IMPORTANT:
The output structures have integer Na occupations.

Occupied Na site:
    Na

Unoccupied Na site:
    removed from the structure

No fractional Na occupations are written into the final structures.
"""

from pathlib import Path
from itertools import combinations
from math import comb

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


# ============================================================
# EXPERIMENTAL TARGET OCCUPATIONS
# ============================================================

TARGET_NA1 = 64.0
TARGET_NA2 = 31.0

# Numerical tolerance for identifying the Na occupations
OCCUPATION_TOLERANCE = 0.02


# ============================================================
# IDENTIFY Na SITES
# ============================================================

def identify_na_sites(structure):
    """
    Identify Na1 and Na2 sites from their fractional occupations.

    Na1:
        occupation approximately 0.64

    Na2:
        occupation approximately 0.31

    Returns
    -------
    dict
        {
            "Na1": [site indices],
            "Na2": [site indices]
        }
    """

    na_groups = {
        "Na1": [],
        "Na2": [],
    }

    for i, site in enumerate(structure):

        # ----------------------------------------------------
        # Only process sites containing Na
        # ----------------------------------------------------

        if "Na" not in site.species:
            continue

        # ----------------------------------------------------
        # Get Na occupation
        # ----------------------------------------------------

        na_occupation = float(site.species["Na"])

        # ----------------------------------------------------
        # Identify Na1
        # ----------------------------------------------------


        if abs(na_occupation - 0.64) <= OCCUPATION_TOLERANCE:

            na_groups["Na1"].append(i)

        # ----------------------------------------------------
        # Identify Na2
        # ----------------------------------------------------

        elif abs(na_occupation - 0.31) <= OCCUPATION_TOLERANCE:

            na_groups["Na2"].append(i)

        # ----------------------------------------------------
        # Unexpected Na occupation
        # ----------------------------------------------------

        else:

            print(
                f"WARNING: Na site {i} has occupation "
                f"{na_occupation:.6f}, which does not match "
                f"Na1 (~0.64) or Na2 (~0.31)."
            )

    # --------------------------------------------------------
    # Make sure both types exist
    # --------------------------------------------------------

    if len(na_groups["Na1"]) == 0:

        raise ValueError(
            "No Na1 sites were found. "
            "Expected Na occupation near 0.64."
        )

    if len(na_groups["Na2"]) == 0:

        raise ValueError(
            "No Na2 sites were found. "
            "Expected Na occupation near 0.31."
        )

    return na_groups


# ============================================================
# CALCULATE INTEGER OCCUPATION
# ============================================================

def calculate_configuration(
    n,
    occurrence,
    target_percent,
):
    """
    Calculate the integer number of occupied sites that gives
    the closest possible occupation to the experimental target.

    N = n x occurrence

    Ideal occupied sites:

        target_percent / 100 x N

    Since the number of occupied sites must be an integer,
    floor and ceiling are checked and the closest is selected.
    """

    # --------------------------------------------------------
    # Total sites after supercell expansion
    # --------------------------------------------------------

    total_sites = n * occurrence

    # --------------------------------------------------------
    # Ideal number of occupied sites
    # --------------------------------------------------------

    ideal_occupied = (
        target_percent / 100.0
    ) * total_sites

    # --------------------------------------------------------
    # Integer possibilities
    # --------------------------------------------------------

    lower = int(ideal_occupied)

    upper = lower + 1

    candidates = []

    if 0 <= lower <= total_sites:

        candidates.append(lower)


    if 0 <= upper <= total_sites:

        candidates.append(upper)

    # --------------------------------------------------------
    # Select integer occupation giving closest percentage
    # --------------------------------------------------------

    occupied = min(
        candidates,
        key=lambda k: abs(
            (k / total_sites) * 100.0
            - target_percent
        )
    )

    # --------------------------------------------------------
    # Number of vacant sites
    # --------------------------------------------------------

    vacant = total_sites - occupied

    # --------------------------------------------------------
    # Actual achievable occupation
    # --------------------------------------------------------

    actual_percent = (
        occupied / total_sites
    ) * 100.0

    # --------------------------------------------------------
    # Difference from experiment
    # --------------------------------------------------------

    difference = abs(
        actual_percent - target_percent
    )

    # --------------------------------------------------------
    # Number of possible arrangements
    #
    # Choose 'occupied' sites from 'total_sites'
    #
    # C(N,k) = N!/[k!(N-k)!]
    # --------------------------------------------------------

    arrangements = comb(
        total_sites,
        occupied
    )

    return {
        "total_sites": total_sites,
        "ideal_occupied": ideal_occupied,
        "occupied": occupied,
        "vacant": vacant,
        "actual_percent": actual_percent,
        "difference": difference,
        "arrangements": arrangements,
    }


# ============================================================
# CREATE SUPERCELL
# ============================================================

def make_supercell(structure, n, direction):
    """
    Create:

        direction 1 -> n x 1 x 1
        direction 2 -> 1 x n x 1
        direction 3 -> 1 x 1 x n
    """

    if direction == 1:

        matrix = [
            [n, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]

    elif direction == 2:

        matrix = [
            [1, 0, 0],
            [0, n, 0],
            [0, 0, 1],
        ]

    elif direction == 3:

        matrix = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, n],
        ]

    else:

        raise ValueError(
            "Direction must be 1, 2, or 3."
        )

    supercell = structure.copy()

    supercell.make_supercell(matrix)

    return supercell


# ============================================================
# GENERATE ALL CONFIGURATIONS
# ============================================================

def generate_configurations(
    structure,
    na_groups,
    occupied_counts,
    output_directory,
):
    """
    Generate every possible Na1 x Na2 configuration.

    For Na1:

        choose k1 sites from N1

    For Na2:

        choose k2 sites from N2

    Every Na1 configuration is combined with every Na2
    configuration.

    Total:

        C(N1,k1) x C(N2,k2)
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Na site indices
    # --------------------------------------------------------

    na1_sites = na_groups["Na1"]
    na2_sites = na_groups["Na2"]

    # --------------------------------------------------------
    # Number of occupied sites required
    # --------------------------------------------------------

    na1_occupied = occupied_counts["Na1"]
    na2_occupied = occupied_counts["Na2"]

    # --------------------------------------------------------
    # Calculate expected numbers
    # --------------------------------------------------------

    number_na1 = comb(
        len(na1_sites),
        na1_occupied
    )

    number_na2 = comb(
        len(na2_sites),
        na2_occupied
    )

    total_expected = (
        number_na1 * number_na2
    )

    print("\n" + "=" * 70)
    print("CONFIGURATION GENERATION")
    print("=" * 70)

    print(
        f"\nNa1 arrangements: {number_na1:,}"
    )

    print(
        f"Na2 arrangements: {number_na2:,}"
    )

    print(
        f"Total structures: "
        f"{total_expected:,}"
    )

    print(
        f"\nOutput directory:"
        f"\n{output_directory}"
    )

    # --------------------------------------------------------
    # Generate Na1 combinations
    # --------------------------------------------------------

    na1_combinations = combinations(
        na1_sites,
        na1_occupied
    )

    structure_number = 1

    # --------------------------------------------------------
    # Combine every Na1 configuration with every Na2
    # configuration
    # --------------------------------------------------------

    for na1_occupied_sites in na1_combinations:

        na1_occupied_set = set(
            na1_occupied_sites
        )

        # A fresh iterator is required because combinations
        # is an iterator.
        na2_combinations = combinations(
            na2_sites,
            na2_occupied
        )

        for na2_occupied_sites in na2_combinations:

            na2_occupied_set = set(
                na2_occupied_sites
            )

            # ------------------------------------------------
            # All occupied Na sites
            # ------------------------------------------------

            occupied_sites = (
                na1_occupied_set
                | na2_occupied_set
            )

            # ------------------------------------------------
            # Find all Na sites that must be removed
            # ------------------------------------------------

            remove_indices = []

            for label in ["Na1", "Na2"]:

                for index in na_groups[label]:

                    if index not in occupied_sites:

                        remove_indices.append(index)

            # ------------------------------------------------
            # Remove from highest index to lowest
            # ------------------------------------------------

            remove_indices.sort(
                reverse=True
            )

            new_structure = structure.copy()

            for index in remove_indices:

                new_structure.remove_sites(
                    [index]
                )

            # ------------------------------------------------
            # Write CIF
            # ------------------------------------------------

            filename = (
                output_directory
                / f"structure_{structure_number:06d}.cif"
            )

            CifWriter(
                new_structure
            ).write_file(filename)

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                structure_number == 1
                or structure_number % 100 == 0
                or structure_number == total_expected
            ):

                print(
                    f"Generated "
                    f"{structure_number:,} / "
                    f"{total_expected:,}"
                )

            structure_number += 1

    print("\nGeneration complete.")

    print(
        f"Total CIF files generated: "
        f"{structure_number - 1:,}"
    )


# ============================================================
# PRINT ANALYSIS
# ============================================================

def print_analysis(
    n,
    na_groups,
):
    """
    Print Na1 and Na2 occupation analysis for a given n.
    """

    na1_occurrence = len(
        na_groups["Na1"]
    )

    na2_occurrence = len(
        na_groups["Na2"]
    )

    na1_result = calculate_configuration(
        n,
        na1_occurrence,
        TARGET_NA1,
    )

    na2_result = calculate_configuration(
        n,
        na2_occurrence,
        TARGET_NA2,
    )

    total_arrangements = (
        na1_result["arrangements"]
        * na2_result["arrangements"]
    )

    print("\n" + "-" * 70)
    print(f"n = {n}")
    print("-" * 70)

    # --------------------------------------------------------
    # Na1
    # --------------------------------------------------------

    print("\nNa1")
    print(
        f"  Experimental occupation : "
        f"{TARGET_NA1:.2f}%"
    )

    print(
        f"  Original occurrences    : "
        f"{na1_occurrence}"
    )

    print(
        f"  Total sites after n     : "
        f"{na1_result['total_sites']}"
    )

    print(
        f"  Ideal occupied sites    : "
        f"{na1_result['ideal_occupied']:.4f}"
    )

    print(
        f"  Integer occupied sites  : "
        f"{na1_result['occupied']}"
    )

    print(
        f"  Vacant sites            : "
        f"{na1_result['vacant']}"
    )

    print(
        f"  Actual occupation       : "
        f"{na1_result['actual_percent']:.4f}%"
    )

    print(
        f"  Difference from target  : "
        f"{na1_result['difference']:.4f}%"
    )

    print(
        f"  Arrangements            : "
        f"{na1_result['arrangements']:,}"
    )

    # --------------------------------------------------------
    # Na2
    # --------------------------------------------------------

    print("\nNa2")
    print(
        f"  Experimental occupation : "
        f"{TARGET_NA2:.2f}%"
    )

    print(
        f"  Original occurrences    : "
        f"{na2_occurrence}"
    )

    print(
        f"  Total sites after n     : "
        f"{na2_result['total_sites']}"
    )

    print(
        f"  Ideal occupied sites    : "
        f"{na2_result['ideal_occupied']:.4f}"
    )

    print(
        f"  Integer occupied sites  : "
        f"{na2_result['occupied']}"
    )

    print(
        f"  Vacant sites            : "
        f"{na2_result['vacant']}"
    )

    print(
        f"  Actual occupation       : "
        f"{na2_result['actual_percent']:.4f}%"
    )

    print(
        f"  Difference from target  : "
        f"{na2_result['difference']:.4f}%"
    )

    print(
        f"  Arrangements            : "
        f"{na2_result['arrangements']:,}"
    )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    print("\nTOTAL")
    print(
        f"  Na1 x Na2 arrangements : "
        f"{total_arrangements:,}"
    )

    return na1_result, na2_result, total_arrangements


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(" ORDERED CIF -> INTEGER Na CONFIGURATIONS")
    print("=" * 70)

    # ========================================================
    # STEP 1: CIF FILE
    # ========================================================

    cif_file = input(
        "\nEnter the path to the ordered CIF file: "
    ).strip()

    cif_path = Path(cif_file)

    if not cif_path.exists():

        raise FileNotFoundError(
            f"\nCIF file not found:\n{cif_path}"
        )

    # ========================================================
    # READ CIF
    # ========================================================

    print(
        f"\nReading:\n{cif_path}"
    )

    structure = Structure.from_file(
        cif_path
    )

    print(
        f"\nTotal sites in original structure: "
        f"{len(structure)}"
    )

    # ========================================================
    # IDENTIFY Na TYPES
    # ========================================================

    na_groups = identify_na_sites(
        structure
    )

    # ========================================================
    # PRINT Na INFORMATION
    # ========================================================

    print("\n" + "=" * 70)
    print("SODIUM SITES FOUND")
    print("=" * 70)

    print("\nNa1:")

    print(
        f"  Experimental occupation : "
        f"{TARGET_NA1:.2f}%"
    )

    print(
        f"  CIF occupation          : "
        f"approximately 0.64"
    )

    print(
        f"  Number of occurrences   : "
        f"{len(na_groups['Na1'])}"
    )

    print("\nNa2:")

    print(
        f"  Experimental occupation : "
        f"{TARGET_NA2:.2f}%"
    )

    print(
        f"  CIF occupation          : "
        f"approximately 0.31"
    )

    print(
        f"  Number of occurrences   : "
        f"{len(na_groups['Na2'])}"
    )

    # ========================================================
    # STEP 2: MAXIMUM n
    # ========================================================

    print("\n" + "=" * 70)
    print("SUPERCELL OCCUPATION ANALYSIS")
    print("=" * 70)

    max_n = int(
        input(
            "\nEnter the maximum supercell multiplier "
            "to test (e.g. 4): "
        )
    )

    if max_n < 1:

        raise ValueError(
            "Maximum n must be at least 1."
        )

    # ========================================================
    # ANALYZE n = 1 ... max_n
    # ========================================================

    analysis_results = {}

    for n in range(1, max_n + 1):

        na1_result, na2_result, total = (
            print_analysis(
                n,
                na_groups
            )
        )

        analysis_results[n] = {
            "Na1": na1_result,
            "Na2": na2_result,
            "total": total,
        }

    # ========================================================
    # FIND BEST n
    # ========================================================

    best_n = min(
        analysis_results,
        key=lambda n:
        (
            analysis_results[n]["Na1"]["difference"]
            +
            analysis_results[n]["Na2"]["difference"]
        )
    )

    print("\n" + "=" * 70)
    print("BEST n")
    print("=" * 70)

    print(
        f"\nBest n based on combined occupation error: "
        f"{best_n}"
    )

    print(
        f"Na1: "
        f"{analysis_results[best_n]['Na1']['actual_percent']:.4f}% "
        f"(target = 64%)"
    )

    print(
        f"Na2: "
        f"{analysis_results[best_n]['Na2']['actual_percent']:.4f}% "
        f"(target = 31%)"
    )

    print(
        f"Total possible configurations: "
        f"{analysis_results[best_n]['total']:,}"
    )

    # ========================================================
    # STEP 3: SELECT n
    # ========================================================

    selected_n = int(
        input(
            f"\nEnter the n you want to generate "
            f"(1-{max_n}): "
        )
    )

    if selected_n not in analysis_results:

        raise ValueError(
            "Invalid n."
        )

    # ========================================================
    # STEP 4: SELECT DIRECTION
    # ========================================================

    print("\n" + "=" * 70)
    print("SUPERCELL DIRECTION")
    print("=" * 70)

    print(
        "\n1 -> n x 1 x 1"
    )

    print(
        "2 -> 1 x n x 1"
    )

    print(
        "3 -> 1 x 1 x n"
    )

    direction = int(
        input(
            "\nEnter direction (1, 2, or 3): "
        )
    )

    if direction not in [1, 2, 3]:

        raise ValueError(
            "Direction must be 1, 2, or 3."
        )

    # ========================================================
    # CREATE SUPERCELL
    # ========================================================

    supercell = make_supercell(
        structure,
        selected_n,
        direction,
    )

    if direction == 1:

        supercell_description = (
            f"{selected_n} x 1 x 1"
        )

    elif direction == 2:

        supercell_description = (
            f"1 x {selected_n} x 1"
        )

    else:

        supercell_description = (
            f"1 x 1 x {selected_n}"
        )

    print("\n" + "=" * 70)
    print("SUPERCELL CREATED")
    print("=" * 70)

    print(
        f"\nSupercell: "
        f"{supercell_description}"
    )

    print(
        f"Total sites in supercell: "
        f"{len(supercell)}"
    )

    # ========================================================
    # IDENTIFY Na SITES AGAIN
    # ========================================================

    supercell_na_groups = (
        identify_na_sites(
            supercell
        )
    )

    print("\nNa sites in supercell:")

    print(
        f"  Na1: "
        f"{len(supercell_na_groups['Na1'])}"
    )

    print(
        f"  Na2: "
        f"{len(supercell_na_groups['Na2'])}"
    )

    # ========================================================
    # GET INTEGER OCCUPATIONS
    # ========================================================

    selected_results = (
        analysis_results[selected_n]
    )

    na1_occupied = (
        selected_results["Na1"]["occupied"]
    )

    na2_occupied = (
        selected_results["Na2"]["occupied"]
    )

    print("\n" + "=" * 70)
    print("INTEGER OCCUPATIONS TO GENERATE")
    print("=" * 70)

    print(
        f"\nNa1:"
        f"\n  Occupied = {na1_occupied}"
        f"\n  Total    = {selected_results['Na1']['total_sites']}"
        f"\n  Actual   = "
        f"{selected_results['Na1']['actual_percent']:.4f}%"
    )

    print(
        f"\nNa2:"
        f"\n  Occupied = {na2_occupied}"
        f"\n  Total    = {selected_results['Na2']['total_sites']}"
        f"\n  Actual   = "
        f"{selected_results['Na2']['actual_percent']:.4f}%"
    )

    # ========================================================
    # CHECK TOTAL NUMBER BEFORE GENERATION
    # ========================================================

    total_structures = (
        selected_results["total"]
    )

    print("\n" + "=" * 70)
    print("GENERATION WARNING")
    print("=" * 70)

    print(
        f"\nThe script is about to generate:"
    )

    print(
        f"{total_structures:,} CIF files."
    )

    print(
        "\nEach CIF represents a different explicit "
        "integer Na1/Na2 configuration."
    )

    if total_structures > 100000:

        print(
            "\nWARNING:"
            "\nThis is a very large number of structures."
            "\nGenerating and storing all of them may consume "
            "a significant amount of disk space."
        )

    proceed = input(
        "\nProceed with generation? (y/n): "
    ).strip().lower()

    if proceed != "y":

        print(
            "\nGeneration cancelled."
        )

        return

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_directory = (
        cif_path.parent
        / (
            f"{cif_path.stem}"
            f"_n{selected_n}"
            f"_{supercell_description.replace(' ', '')}"
            f"_integer_configs"
        )
    )

    # ========================================================
    # GENERATE CONFIGURATIONS
    # ========================================================

    occupied_counts = {
        "Na1": na1_occupied,
        "Na2": na2_occupied,
    }

    generate_configurations(
        supercell,
        supercell_na_groups,
        occupied_counts,
        output_directory,
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n" + "=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(
        f"\nCIF files are located in:"
        f"\n{output_directory}"
    )

    print(
        "\nThese structures contain integer Na occupations "
        "and are ready for further structural inspection "
        "and DFT preparation."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

