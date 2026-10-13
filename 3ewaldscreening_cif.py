"""
Symmetry reduction and Ewald-energy screening of ordered CIF configurations.

Workflow
--------
1. Read ordered CIF files.
2. Identify structurally equivalent configurations.
3. Retain representative CIFs and record their original configurations.
4. Calculate the charge on the selected variable element for each structure.
5. Calculate total Ewald energy for every retained structure.
6. Rank structures by total Ewald energy, lowest first.
7. Group structures whose energies are within 1 meV of a class representative.
8. Generate a connected graph of the energy classes.
9. Save CSV tables, CIFs, and screening settings.

Important:
Ewald energy is a point-charge electrostatic descriptor. It is not a
substitute for DFT total energies or a complete thermodynamic stability test.
"""

import csv
import math
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pymatgen.core import Species, Structure
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.structure_matcher import (
    StructureMatcher,
    ElementComparator,
)



VARIABLE_ELEMENT = "V"

# Formal charges for all elements except the variable element.
FIXED_CHARGES = {
    "Na": 1.0,
    "O": -2.0,
}

# Energy-class tolerance: 1 meV = 0.001 eV.
ENERGY_TOLERANCE_EV = 0.001

# Structural matching tolerances.
LATTICE_LENGTH_TOL = 0.01
LATTICE_ANGLE_TOL = 0.5
SITE_TOL = 0.01

# Set False to disable structural deduplication.
APPLY_SYMMETRY_REDUCTION = True

# Print progress at approximately these intervals.
PROGRESS_INTERVAL = 100


def stage(message):
    print("\n" + "=" * 72, flush=True)
    print(message, flush=True)
    print("=" * 72, flush=True)


def report_progress(label, completed, total, start_time):
    elapsed = time.time() - start_time

    print(
        f"[{label}] {completed}/{total} completed "
        f"| elapsed: {elapsed:.1f} s",
        flush=True,
    )


def ask_directory(prompt, default=None):
    if default is not None:
        answer = input(f"{prompt} [{default}]: ").strip()
        answer = answer or str(default)
    else:
        answer = input(f"{prompt}: ").strip()

    path = Path(answer).expanduser().resolve()

    if not path.is_dir():
        raise NotADirectoryError(
            f"Directory does not exist: {path}"
        )

    return path


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def composition_signature(structure):
    """
    Return a composition signature independent of site ordering.
    """
    amounts = structure.composition.get_el_amt_dict()

    return tuple(
        sorted(
            (element, round(float(amount), 6))
            for element, amount in amounts.items()
        )
    )


def lattice_signature(structure):
    """
    Create a coarse lattice signature to reduce the number of
    unnecessary structural comparisons.

    Sorting lattice lengths helps accommodate equivalent cells
    whose axes are permuted.
    """
    lengths = tuple(
        sorted(round(float(x), 2) for x in structure.lattice.abc)
    )

    angles = tuple(
        sorted(round(float(x), 1) for x in structure.lattice.angles)
    )

    return (
        composition_signature(structure),
        len(structure),
        lengths,
        angles,
    )


# ============================================================
# STRUCTURE-SPECIFIC CHARGE CALCULATION
# ============================================================


def calculate_variable_charge(structure):
    """
    Calculate the average charge on VARIABLE_ELEMENT from charge neutrality.

    Sum(n_i * q_i) = 0

    q_variable = -sum(n_fixed * q_fixed) / n_variable
    """

    amounts = structure.composition.get_el_amt_dict()

    if VARIABLE_ELEMENT not in amounts:
        raise ValueError(
            f"The structure does not contain {VARIABLE_ELEMENT}."
        )

    n_variable = float(amounts[VARIABLE_ELEMENT])

    if math.isclose(n_variable, 0.0, abs_tol=1e-12):
        raise ValueError(
            f"No {VARIABLE_ELEMENT} atoms found."
        )

    fixed_charge_sum = 0.0

    for element, amount in amounts.items():

        if element == VARIABLE_ELEMENT:
            continue

        if element not in FIXED_CHARGES:
            raise ValueError(
                f"No fixed charge specified for {element}. "
                f"Add it to FIXED_CHARGES before continuing."
            )

        fixed_charge_sum += (
            float(amount) * float(FIXED_CHARGES[element])
        )

    return -fixed_charge_sum / n_variable


def assign_oxidation_states(structure, variable_charge):
    """
    Make a charged copy of the structure for EwaldSummation.

    The original structure is not modified.
    """

    species = []

    for site in structure:

        element = site.specie.symbol

        if element == VARIABLE_ELEMENT:
            charge = variable_charge

        elif element in FIXED_CHARGES:
            charge = FIXED_CHARGES[element]

        else:
            raise ValueError(
                f"No charge assigned to element {element}."
            )

        species.append(
            Species(element, oxidation_state=float(charge))
        )

    charged_structure = Structure(
        lattice=structure.lattice,
        species=species,
        coords=structure.frac_coords,
        coords_are_cartesian=False,
        to_unit_cell=True,
        site_properties=structure.site_properties,
    )

    return charged_structure


# ============================================================
# SYMMETRY / STRUCTURAL DEDUPLICATION
# ============================================================


def reduce_structures(paths, structures):
    """
    Group structurally equivalent configurations and retain one
    representative from each group.

    StructureMatcher is used to compare equivalent structures.
    The input configurations are assumed to use the same selected
    supercell, so primitive-cell and supercell transformations are
    disabled.

    Comparisons are restricted to coarse composition/lattice buckets
    for performance.
    """

    matcher = StructureMatcher(
        ltol=LATTICE_LENGTH_TOL,
        stol=SITE_TOL,
        angle_tol=LATTICE_ANGLE_TOL,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
        comparator=ElementComparator(),
    )

    buckets = {}

    for index, structure in enumerate(structures):
        key = lattice_signature(structure)
        buckets.setdefault(key, []).append(index)

    groups = []

    start_time = time.time()
    total = len(structures)
    processed = 0

    for bucket_indices in buckets.values():

        bucket_groups = []

        for index in bucket_indices:

            matched_group = None

            for group in bucket_groups:

                representative_index = group[0]

                try:
                    equivalent = matcher.fit(
                        structures[index],
                        structures[representative_index],
                    )
                except Exception:
                    equivalent = False

                if equivalent:
                    matched_group = group
                    break

            if matched_group is None:
                bucket_groups.append([index])
            else:
                matched_group.append(index)

            processed += 1

            if (
                processed == total
                or processed % PROGRESS_INTERVAL == 0
            ):
                report_progress(
                    "Symmetry reduction",
                    processed,
                    total,
                    start_time,
                )

        groups.extend(bucket_groups)

    representative_indices = []
    mapping_rows = []
    summary_rows = []

    for group_number, group in enumerate(groups, start=1):

        representative_index = group[0]
        representative_path = paths[representative_index]

        representative_id = f"SYM{group_number:06d}"

        representative_indices.append(representative_index)

        summary_rows.append({
            "representative_id": representative_id,
            "representative_file": representative_path.name,
            "number_of_original_structures": len(group),
            "equivalent_original_files": "; ".join(
                paths[i].name for i in group
            ),
        })

        for index in group:

            mapping_rows.append({
                "original_file": paths[index].name,
                "representative_id": representative_id,
                "representative_file": representative_path.name,
                "is_representative": (
                    index == representative_index
                ),
            })

    return (
        representative_indices,
        summary_rows,
        mapping_rows,
    )


# ============================================================
# ENERGY CLASSIFICATION
# ============================================================


def group_energy_classes(ranked_rows):
    """
    Group structures whose energies differ by no more than the
    tolerance from the lowest-energy member of that class.

    This avoids chained grouping. For example, three structures
    separated successively by 0.8 meV are not automatically placed
    in the same class if the third is more than 1 meV from the first.
    """

    classes = []

    for row in ranked_rows:

        energy = float(row["ewald_energy_eV"])

        if not classes:

            classes.append({
                "class_id": "EC0001",
                "representative_energy_eV": energy,
                "members": [row],
            })

            continue

        current_class = classes[-1]

        difference = (
            energy - current_class["representative_energy_eV"]
        )

        if difference <= ENERGY_TOLERANCE_EV + 1e-12:

            current_class["members"].append(row)

        else:

            classes.append({
                "class_id": f"EC{len(classes) + 1:04d}",
                "representative_energy_eV": energy,
                "members": [row],
            })

    class_rows = []

    for class_rank, energy_class in enumerate(classes, start=1):

        members = energy_class["members"]

        class_rows.append({
            "class_rank": class_rank,
            "class_id": energy_class["class_id"],
            "representative_energy_eV": (
                f'{energy_class["representative_energy_eV"]:.9f}'
            ),
            "energy_tolerance_eV": (
                f"{ENERGY_TOLERANCE_EV:.6f}"
            ),
            "number_of_structures": len(members),
            "individual_ranks": "; ".join(
                str(member["rank"]) for member in members
            ),
            "representative_cif": members[0]["cif_file"],
            "member_cif_files": "; ".join(
                member["cif_file"] for member in members
            ),
            "member_energies_eV": "; ".join(
                f'{float(member["ewald_energy_eV"]):.9f}'
                for member in members
            ),
        })

    return class_rows


# ============================================================
# CONNECTED GRAPH
# ============================================================


def plot_energy_classes(class_rows, output_png):
    """
    Create a static connected graph with one point per energy class.
    """

    ranks = [
        int(row["class_rank"])
        for row in class_rows
    ]

    energies = [
        float(row["representative_energy_eV"])
        for row in class_rows
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        ranks,
        energies,
        marker="o",
        linestyle="-",
        linewidth=1.4,
        markersize=4.5,
        label="Energy-class representative",
    )

    ax.set_title(
        "Ewald Energy Ranking by Energy Class",
        fontsize=14,
        pad=14,
    )

    ax.set_xlabel(
        "Energy-class rank (lowest energy = rank 1)",
        fontsize=11,
    )

    ax.set_ylabel(
        "Total Ewald energy (eV)",
        fontsize=11,
    )

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.legend(loc="best")

    ax.ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False,
    )

    if len(ranks) <= 40:
        ax.set_xticks(ranks)

    fig.tight_layout()

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN WORKFLOW
# ============================================================


def main():

    print("\nSYMMETRY REDUCTION AND EWALD SCREENING")
    print("Rank 1 = lowest total Ewald energy.")
    print(
        f"Energy-class tolerance = "
        f"{ENERGY_TOLERANCE_EV * 1000:.3f} meV"
    )

    try:

        input_dir = ask_directory(
            "Enter the directory containing ordered CIF files"
        )

        default_output = (
            input_dir.parent /
            f"{input_dir.name}_ewald_results"
        )

        output_answer = input(
            f"Enter output directory [{default_output}]: "
        ).strip()

        output_dir = Path(
            output_answer or str(default_output)
        ).expanduser().resolve()

    except (ValueError, OSError, KeyboardInterrupt) as exc:

        print(f"\nERROR: {exc}")
        return

    if input_dir == output_dir:
        print("ERROR: The output directory must differ from the input.")
        return

    cif_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".cif"
    )

    if not cif_paths:
        print(f"ERROR: No CIF files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    representative_dir = output_dir / "symmetry_representatives"
    ranked_dir = output_dir / "ranked_representative_cifs"

    representative_dir.mkdir(exist_ok=True)
    ranked_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # STAGE 1: Read CIF files
    # --------------------------------------------------------

    stage("STAGE 1/7 — READING CIF FILES")

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Input CIF count: {len(cif_paths)}")

    valid_paths = []
    structures = []
    failed_rows = []

    start_time = time.time()

    for number, path in enumerate(cif_paths, start=1):

        try:

            structure = Structure.from_file(str(path))

            if len(structure) == 0:
                raise ValueError("Structure contains no sites.")

            valid_paths.append(path)
            structures.append(structure)

        except Exception as exc:

            failed_rows.append({
                "cif_file": path.name,
                "error": str(exc),
            })

        if (
            number == len(cif_paths)
            or number % PROGRESS_INTERVAL == 0
        ):
            report_progress(
                "Reading CIFs",
                number,
                len(cif_paths),
                start_time,
            )

    if failed_rows:

        write_csv(
            output_dir / "failed_cif_files.csv",
            ["cif_file", "error"],
            failed_rows,
        )

        print(
            f"WARNING: {len(failed_rows)} CIF files failed to load. "
            "See failed_cif_files.csv."
        )

    if not structures:
        print("ERROR: No valid CIF files were read.")
        return

    # --------------------------------------------------------
    # STAGE 2: Symmetry reduction
    # --------------------------------------------------------

    stage("STAGE 2/7 — SYMMETRY-BASED STRUCTURAL REDUCTION")

    if APPLY_SYMMETRY_REDUCTION:

        (
            representative_indices,
            symmetry_summary,
            symmetry_mapping,
        ) = reduce_structures(valid_paths, structures)

    else:

        representative_indices = list(range(len(structures)))

        symmetry_summary = [
            {
                "representative_id": f"SYM{i + 1:06d}",
                "representative_file": path.name,
                "number_of_original_structures": 1,
                "equivalent_original_files": path.name,
            }
            for i, path in enumerate(valid_paths)
        ]

        symmetry_mapping = [
            {
                "original_file": path.name,
                "representative_id": f"SYM{i + 1:06d}",
                "representative_file": path.name,
                "is_representative": True,
            }
            for i, path in enumerate(valid_paths)
        ]

    representative_paths = [
        valid_paths[i] for i in representative_indices
    ]

    representative_structures = [
        structures[i] for i in representative_indices
    ]

    write_csv(
        output_dir / "symmetry_reduction_summary.csv",
        [
            "representative_id",
            "representative_file",
            "number_of_original_structures",
            "equivalent_original_files",
        ],
        symmetry_summary,
    )

    write_csv(
        output_dir / "symmetry_mapping.csv",
        [
            "original_file",
            "representative_id",
            "representative_file",
            "is_representative",
        ],
        symmetry_mapping,
    )

    for path in representative_paths:
        shutil.copy2(
            path,
            representative_dir / path.name,
        )

    print(f"Valid original CIFs: {len(valid_paths)}")
    print(f"Unique representatives: {len(representative_paths)}")
    print(
        "Equivalent configurations removed: "
        f"{len(valid_paths) - len(representative_paths)}"
    )

    print(
        "Symmetry summary: "
        f"{output_dir / 'symmetry_reduction_summary.csv'}"
    )

    print(
        "Original-to-representative mapping: "
        f"{output_dir / 'symmetry_mapping.csv'}"
    )

    # --------------------------------------------------------
    # STAGE 3: Calculate charge for each structure
    # --------------------------------------------------------

    stage(
        "STAGE 3/7 — CALCULATING STRUCTURE-SPECIFIC CHARGES"
    )

    charge_rows = []
    charge_data = []

    start_time = time.time()
    total = len(representative_structures)

    for number, (path, structure) in enumerate(
        zip(representative_paths, representative_structures),
        start=1,
    ):

        try:

            variable_charge = calculate_variable_charge(structure)

            charged_structure = assign_oxidation_states(
                structure,
                variable_charge,
            )

            charge_data.append({
                "path": path,
                "structure": structure,
                "charged_structure": charged_structure,
                "variable_charge": variable_charge,
            })

            charge_rows.append({
                "cif_file": path.name,
                "formula": structure.composition.reduced_formula,
                "number_of_sites": len(structure),
                "variable_element": VARIABLE_ELEMENT,
                "calculated_variable_element_charge": (
                    f"{variable_charge:.9f}"
                ),
                "fixed_charges": str(FIXED_CHARGES),
                "status": "OK",
                "error": "",
            })

        except Exception as exc:

            charge_rows.append({
                "cif_file": path.name,
                "formula": structure.composition.reduced_formula,
                "number_of_sites": len(structure),
                "variable_element": VARIABLE_ELEMENT,
                "calculated_variable_element_charge": "",
                "fixed_charges": str(FIXED_CHARGES),
                "status": "FAILED",
                "error": str(exc),
            })

            print(f"WARNING: {path.name}: {exc}")

        if (
            number == total
            or number % PROGRESS_INTERVAL == 0
        ):
            report_progress(
                "Charge calculations",
                number,
                total,
                start_time,
            )

    write_csv(
        output_dir / "calculated_charges.csv",
        [
            "cif_file",
            "formula",
            "number_of_sites",
            "variable_element",
            "calculated_variable_element_charge",
            "fixed_charges",
            "status",
            "error",
        ],
        charge_rows,
    )

    print(
        f"Structures with valid charges: {len(charge_data)}"
    )

    if not charge_data:
        print(
            "ERROR: No valid charge assignments. "
            "Check FIXED_CHARGES and calculated_charges.csv."
        )
        return

    # --------------------------------------------------------
    # STAGE 4: Calculate Ewald energies and rank
    # --------------------------------------------------------

    stage("STAGE 4/7 — CALCULATING TOTAL EWALD ENERGIES")

    energy_rows = []
    start_time = time.time()
    total = len(charge_data)

    for number, record in enumerate(charge_data, start=1):

        path = record["path"]
        structure = record["structure"]
        charged_structure = record["charged_structure"]
        variable_charge = record["variable_charge"]

        try:

            ewald_energy = float(
                EwaldSummation(charged_structure).total_energy
            )

            energy_rows.append({
                "cif_file": path.name,
                "formula": structure.composition.reduced_formula,
                "calculated_variable_element_charge": (
                    variable_charge
                ),
                "ewald_energy_eV": ewald_energy,
                "status": "OK",
                "error": "",
            })

        except Exception as exc:

            energy_rows.append({
                "cif_file": path.name,
                "formula": structure.composition.reduced_formula,
                "calculated_variable_element_charge": (
                    variable_charge
                ),
                "ewald_energy_eV": "",
                "status": "FAILED",
                "error": str(exc),
            })

            print(
                f"WARNING: Ewald calculation failed for "
                f"{path.name}: {exc}"
            )

        if (
            number == total
            or number % PROGRESS_INTERVAL == 0
        ):
            report_progress(
                "Ewald calculations",
                number,
                total,
                start_time,
            )

    successful_rows = [
        row for row in energy_rows
        if row["status"] == "OK"
    ]

    write_csv(
        output_dir / "ewald_calculation_results.csv",
        [
            "cif_file",
            "formula",
            "calculated_variable_element_charge",
            "ewald_energy_eV",
            "status",
            "error",
        ],
        energy_rows,
    )

    if not successful_rows:
        print("ERROR: All Ewald calculations failed.")
        return

    # Rank by total Ewald energy: most negative is rank 1.
    successful_rows.sort(
        key=lambda row: float(row["ewald_energy_eV"])
    )

    for rank, row in enumerate(successful_rows, start=1):
        row["rank"] = rank

    ranked_csv_rows = [
        {
            **row,
            "ewald_energy_eV": (
                f'{float(row["ewald_energy_eV"]):.9f}'
            ),
            "calculated_variable_element_charge": (
                f'{float(row["calculated_variable_element_charge"]):.9f}'
            ),
        }
        for row in successful_rows
    ]

    write_csv(
        output_dir / "ewald_ranked_structures.csv",
        [
            "rank",
            "cif_file",
            "formula",
            "calculated_variable_element_charge",
            "ewald_energy_eV",
            "status",
            "error",
        ],
        ranked_csv_rows,
    )

    print(f"Successful Ewald calculations: {len(successful_rows)}")

    print(
        "Lowest Ewald energy: "
        f'{float(successful_rows[0]["ewald_energy_eV"]):.9f} eV'
    )

    print(
        "Highest Ewald energy: "
        f'{float(successful_rows[-1]["ewald_energy_eV"]):.9f} eV'
    )

    # --------------------------------------------------------
    # STAGE 5: Group energies
    # --------------------------------------------------------

    stage("STAGE 5/7 — GROUPING STRUCTURES WITHIN 1 meV")

    class_rows = group_energy_classes(successful_rows)

    write_csv(
        output_dir / "ewald_energy_classes.csv",
        [
            "class_rank",
            "class_id",
            "representative_energy_eV",
            "energy_tolerance_eV",
            "number_of_structures",
            "individual_ranks",
            "representative_cif",
            "member_cif_files",
            "member_energies_eV",
        ],
        class_rows,
    )

    print(f"Ranked structures: {len(successful_rows)}")
    print(f"Energy classes: {len(class_rows)}")
    print(
        "Class table: "
        f"{output_dir / 'ewald_energy_classes.csv'}"
    )

    # --------------------------------------------------------
    # STAGE 6: Generate connected graph
    # --------------------------------------------------------

    stage("STAGE 6/7 — GENERATING CONNECTED ENERGY-CLASS GRAPH")

    graph_path = output_dir / "ewald_energy_class_ranking.png"

    plot_energy_classes(
        class_rows,
        graph_path,
    )

    print(f"Graph saved to: {graph_path}")

    # --------------------------------------------------------
    # STAGE 7: Save ranked CIFs and settings
    # --------------------------------------------------------

    stage("STAGE 7/7 — SAVING RANKED CIF FILES")

    rank_by_filename = {
        row["cif_file"]: row["rank"]
        for row in successful_rows
    }

    for record in charge_data:

        path = record["path"]

        if path.name not in rank_by_filename:
            continue

        rank = rank_by_filename[path.name]

        destination = ranked_dir / (
            f"rank_{rank:06d}_{path.name}"
        )

        shutil.copy2(path, destination)

    settings_path = output_dir / "screening_settings.txt"

    with open(settings_path, "w", encoding="utf-8") as file:

        file.write("Symmetry reduction and Ewald screening settings\n")
        file.write(f"Input directory: {input_dir}\n")
        file.write(f"Output directory: {output_dir}\n")
        file.write(f"Variable element: {VARIABLE_ELEMENT}\n")
        file.write(f"Fixed charges: {FIXED_CHARGES}\n")
        file.write(
            f"Energy tolerance (eV): {ENERGY_TOLERANCE_EV}\n"
        )
        file.write(
            f"Symmetry reduction enabled: {APPLY_SYMMETRY_REDUCTION}\n"
        )
        file.write(f"Lattice length tolerance: {LATTICE_LENGTH_TOL}\n")
        file.write(f"Lattice angle tolerance: {LATTICE_ANGLE_TOL}\n")
        file.write(f"Site tolerance: {SITE_TOL}\n")
        file.write(f"Valid original CIFs: {len(valid_paths)}\n")
        file.write(
            f"Symmetry representatives: {len(representative_paths)}\n"
        )
        file.write(
            f"Successful Ewald calculations: {len(successful_rows)}\n"
        )
        file.write(f"Energy classes: {len(class_rows)}\n")
        file.write(
            "\nCaution: Ewald energy is a point-charge electrostatic "
            "screening descriptor, not a DFT total energy.\n"
        )

    stage("EWALD SCREENING COMPLETE")

    print(f"Valid input CIFs: {len(valid_paths)}")
    print(f"Symmetry representatives: {len(representative_paths)}")
    print(f"Successful Ewald calculations: {len(successful_rows)}")
    print(f"Energy classes: {len(class_rows)}")
    print(f"\nResults directory: {output_dir}")

    print("\nMAIN OUTPUT FILES")
    print("  symmetry_reduction_summary.csv")
    print("  symmetry_mapping.csv")
    print("  calculated_charges.csv")
    print("  ewald_calculation_results.csv")
    print("  ewald_ranked_structures.csv")
    print("  ewald_energy_classes.csv")
    print("  ewald_energy_class_ranking.png")
    print("  screening_settings.txt")
    print("  symmetry_representatives/")
    print("  ranked_representative_cifs/")

    print(
        "\nScientific reminder: low Ewald energy is a screening "
        "signal, not proof of DFT stability."
    )


if __name__ == "__main__":
    main()
