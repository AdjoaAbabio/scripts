from pathlib import Path
from itertools import combinations, product
from math import comb
import os

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


OCCUPATION_TOLERANCE = 0.001


def identify_na_sites(structure):
    groups = []

    for index, site in enumerate(structure):
        if "Na" not in site.species:
            continue

        na_occupation = float(site.species["Na"])

        found_group = False

        for group in groups:
            if abs(group["occupation"] - na_occupation) <= OCCUPATION_TOLERANCE:
                group["indices"].append(index)
                found_group = True
                break

        if not found_group:
            groups.append({
                "occupation": na_occupation,
                "indices": [index]
            })

    groups.sort(
        key=lambda x: x["occupation"],
        reverse=True
    )

    na_groups = {}

    for i, group in enumerate(groups, start=1):
        na_groups[f"Na{i}"] = group

    return na_groups


def identify_supercell_na_sites(supercell, original_na_groups):
    supercell_na_groups = {}

    for name, original_group in original_na_groups.items():
        target_occupation = original_group["occupation"]

        indices = []

        for index, site in enumerate(supercell):
            if "Na" not in site.species:
                continue

            occupation = float(site.species["Na"])

            if abs(occupation - target_occupation) <= OCCUPATION_TOLERANCE:
                indices.append(index)

        supercell_na_groups[name] = {
            "occupation": target_occupation,
            "indices": indices
        }

    return supercell_na_groups


def print_na_information(na_groups):
    print("\n" + "=" * 70)
    print("SODIUM SITE INFORMATION")
    print("=" * 70)

    total_na_sites = 0

    for name, group in na_groups.items():
        occupation = group["occupation"]
        indices = group["indices"]
        occurrence = len(indices)

        total_na_sites += occurrence

        print(f"\n{name}")
        print(f"  Occupation : {occupation * 100:.4f}%")
        print(f"  Occurrence : {occurrence}")
        print(f"  Site indices : {indices}")

    print("\n" + "-" * 70)
    print(f"Total Na sites in CIF: {total_na_sites}")


def calculate_configuration(n, occurrence, target_percent):
    total_sites = n * occurrence

    ideal_occupied = (
        target_percent / 100.0
    ) * total_sites

    lower = int(ideal_occupied)
    upper = lower + 1

    occupation_increment = 100.0 / total_sites

    candidates = []

    if 0 <= lower <= total_sites:
        candidates.append(lower)

    if 0 <= upper <= total_sites:
        candidates.append(upper)

    occupied = min(
        candidates,
        key=lambda k: abs(
            (k / total_sites) * 100.0
            - target_percent
        )
    )

    vacant = total_sites - occupied

    actual_percent = (
        occupied / total_sites
    ) * 100.0

    error = abs(
        actual_percent - target_percent
    )

    number_of_configurations = comb(
        total_sites,
        occupied
    )

    return {
        "total_sites": total_sites,
        "ideal_occupied": ideal_occupied,
        "occupation_increment": occupation_increment,
        "occupied": occupied,
        "vacant": vacant,
        "actual_percent": actual_percent,
        "error": error,
        "number_of_configurations": number_of_configurations,
    }


def print_analysis(structure, n):
    na_groups = identify_na_sites(structure)

    print(f"\n{'=' * 70}")
    print(f"ANALYSIS FOR SUPERCELL n = {n}")
    print(f"{'=' * 70}")

    total_configurations = 1

    for name, group in na_groups.items():
        occurrence = len(group["indices"])
        target_percent = group["occupation"] * 100.0

        result = calculate_configuration(
            n=n,
            occurrence=occurrence,
            target_percent=target_percent
        )

        total_configurations *= result["number_of_configurations"]

        print(f"\n{name}")
        print(f"  Target occupation       : {target_percent:.4f}%")
        print(f"  Original occurrence     : {occurrence}")
        print(f"  Total sites in n={n}    : {result['total_sites']}")
        print(f"  Occupation increment    : {result['occupation_increment']:.4f}%")
        print(f"  Ideal occupied sites    : {result['ideal_occupied']:.4f}")
        print(f"  Occupied sites          : {result['occupied']}")
        print(f"  Vacant sites            : {result['vacant']}")
        print(f"  Actual occupation       : {result['actual_percent']:.4f}%")
        print(f"  Error                   : {result['error']:.4f}%")
        print(f"  Configurations          : {result['number_of_configurations']:,}")

    print(f"\n{'-' * 70}")
    print(f"TOTAL CONFIGURATIONS      : {total_configurations:,}")
    print(f"{'=' * 70}")


def make_supercell(structure, n, direction):
    if direction in ("a", "A", 1):
        matrix = [
            [n, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        direction_label = "a"
        supercell_label = f"{n}x1x1"

    elif direction in ("b", "B", 2):
        matrix = [
            [1, 0, 0],
            [0, n, 0],
            [0, 0, 1]
        ]
        direction_label = "b"
        supercell_label = f"1x{n}x1"

    elif direction in ("c", "C", 3):
        matrix = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, n]
        ]
        direction_label = "c"
        supercell_label = f"1x1x{n}"

    else:
        raise ValueError(
            "Direction must be a, b, or c."
        )

    supercell = structure.copy()
    supercell.make_supercell(matrix)

    return supercell, direction_label, supercell_label


def calculate_supercell_configurations(
    supercell_na_groups,
    target_percentages
):
    configuration_results = {}

    for name, group in supercell_na_groups.items():
        if name not in target_percentages:
            raise ValueError(
                f"Detected unexpected Na group: {name}"
            )

        total_sites = len(group["indices"])

        if total_sites == 0:
            raise ValueError(
                f"No sites found for {name} in the supercell."
            )

        occupation_increment = 100.0 / total_sites
        target_percent = target_percentages[name]

        ideal_occupied = (
            target_percent / occupation_increment
        )

        lower = max(
            0,
            min(int(ideal_occupied), total_sites)
        )

        upper = max(
            0,
            min(lower + 1, total_sites)
        )

        lower_actual = lower * occupation_increment
        upper_actual = upper * occupation_increment

        if abs(lower_actual - target_percent) <= abs(
            upper_actual - target_percent
        ):
            occupied = lower
        else:
            occupied = upper

        vacant = total_sites - occupied
        actual_percent = occupied * occupation_increment
        error = abs(actual_percent - target_percent)

        number_of_configurations = comb(
            total_sites,
            occupied
        )

        configuration_results[name] = {
            "total_sites": total_sites,
            "occupation_increment": occupation_increment,
            "ideal_occupied": ideal_occupied,
            "occupied": occupied,
            "vacant": vacant,
            "actual_percent": actual_percent,
            "error": error,
            "number_of_configurations": number_of_configurations
        }

    return configuration_results


def generate_configurations(
    supercell,
    na_groups,
    configuration_results,
    output_directory
):
    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    group_combinations = []

    for name, group in na_groups.items():
        indices = group["indices"]
        occupied_number = configuration_results[name]["occupied"]

        combinations_for_group = list(
            combinations(
                indices,
                occupied_number
            )
        )

        group_combinations.append(
            combinations_for_group
        )

        print(
            f"{name}: "
            f"{len(combinations_for_group):,} combinations"
        )

    total_expected = 1

    for combinations_for_group in group_combinations:
        total_expected *= len(combinations_for_group)

    print(
        f"\nTotal configurations to generate: "
        f"{total_expected:,}"
    )

    all_na_indices = set()

    for group in na_groups.values():
        all_na_indices.update(group["indices"])

    generated = 0

    for selected_groups in product(*group_combinations):
        generated += 1

        occupied_indices = set()

        for selected in selected_groups:
            occupied_indices.update(selected)

        vacant_indices = (
            all_na_indices - occupied_indices
        )

        new_structure = supercell.copy()

        for index in occupied_indices:
            new_structure.replace(index, "Na")

        for index in sorted(vacant_indices, reverse=True):
            new_structure.remove_sites([index])

        filename = (
            output_directory
            / f"newstructures_{generated:06d}.cif"
        )

        CifWriter(
            new_structure
        ).write_file(filename)

        if (
            generated == 1
            or generated % 1000 == 0
            or generated == total_expected
        ):
            print(
                f"  Generated "
                f"{generated:,}/{total_expected:,}"
            )

    print("\nConfiguration generation complete.")
    print(
        f"Total CIF files generated: "
        f"{generated:,}"
    )


def main():
    print("=" * 70)
    print("ORDERED OCCUPIED CIF → INTEGER Na CONFIGURATIONS")
    print("=" * 70)

    cif_file = input(
        "\nEnter the path to your CIF file: "
    ).strip()

    if not os.path.isfile(cif_file):
        raise FileNotFoundError(
            f"\nCIF file not found:\n{cif_file}"
        )

    print(f"\nReading CIF:\n{cif_file}")

    structure = Structure.from_file(cif_file)

    print(
        f"\nTotal sites in original structure: "
        f"{len(structure)}"
    )

    na_groups = identify_na_sites(structure)

    if not na_groups:
        raise ValueError(
            "\nNo sodium sites were detected in the CIF."
        )

    print_na_information(na_groups)

    print("\nDetected Na groups:")

    for name, group in na_groups.items():
        print(
            f"  {name}: "
            f"{group['occupation'] * 100:.4f}% occupation, "
            f"{len(group['indices'])} occurrence(s)"
        )

    max_n = int(
        input(
            "\nEnter maximum supercell multiplier n to test: "
        )
    )

    target_percentages = {}

    for name, group in na_groups.items():
        target_percentages[name] = (
            group["occupation"] * 100.0
        )

    print("\nTarget occupations taken from CIF:")

    for name, target in target_percentages.items():
        print(
            f"  {name}: {target:.4f}%"
        )

    analysis = {}

    print("\n" + "=" * 70)
    print("SUPERCELL ANALYSIS")
    print("=" * 70)

    for n in range(1, max_n + 1):
        print(f"\n{'-' * 70}")
        print(f"n = {n}")
        print(f"{'-' * 70}")

        analysis[n] = {}

        total_configurations = 1
        total_error = 0.0

        for name, group in na_groups.items():
            result = calculate_configuration(
                n=n,
                occurrence=len(group["indices"]),
                target_percent=target_percentages[name]
            )

            analysis[n][name] = result

            total_configurations *= (
                result["number_of_configurations"]
            )

            total_error += result["error"]

            print(f"\n{name}")
            print(
                f"  Total sites       : "
                f"{result['total_sites']}"
            )
            print(
                f"  Occupation step   : "
                f"{result['occupation_increment']:.6f}%"
            )
            print(
                f"  Ideal occupied    : "
                f"{result['ideal_occupied']:.4f}"
            )
            print(
                f"  Occupied sites    : "
                f"{result['occupied']}"
            )
            print(
                f"  Vacant sites      : "
                f"{result['vacant']}"
            )
            print(
                f"  Actual occupation: "
                f"{result['actual_percent']:.6f}%"
            )
            print(
                f"  Target occupation: "
                f"{target_percentages[name]:.6f}%"
            )
            print(
                f"  Error             : "
                f"{result['error']:.6f}%"
            )
            print(
                f"  Configurations    : "
                f"{result['number_of_configurations']:,}"
            )

        analysis[n]["total_configurations"] = (
            total_configurations
        )

        analysis[n]["total_error"] = total_error

        print(
            f"\nTOTAL CONFIGURATIONS: "
            f"{total_configurations:,}"
        )

        print(
            f"TOTAL OCCUPATION ERROR: "
            f"{total_error:.6f}%"
        )

    best_n = min(
        analysis,
        key=lambda x: analysis[x]["total_error"]
    )

    print("\n" + "=" * 70)
    print("BEST SUPERCELL")
    print("=" * 70)

    print(
        f"\nBest n based on total occupation error: "
        f"{best_n}"
    )

    print(
        f"Total error: "
        f"{analysis[best_n]['total_error']:.6f}%"
    )

    print(
        f"Total configurations: "
        f"{analysis[best_n]['total_configurations']:,}"
    )

    selected_n = int(
        input(
            "\nEnter the n value to use for configuration generation: "
        )
    )

    if selected_n not in analysis:
        raise ValueError(
            f"Invalid n. Choose a value between 1 and {max_n}."
        )

    direction = input(
        "\nChoose supercell direction (a, b, or c): "
    ).strip().lower()

    if direction not in ("a", "b", "c"):
        raise ValueError(
            "Direction must be a, b, or c."
        )

    supercell, direction_label, supercell_label = make_supercell(
        structure,
        selected_n,
        direction
    )

    print(
        f"\nCreating {supercell_label} "
        f"along {direction_label}-direction..."
    )

    print(
        f"Original structure sites : {len(structure)}"
    )

    print(
        f"Supercell sites          : {len(supercell)}"
    )

    supercell_na_groups = identify_supercell_na_sites(
        supercell,
        na_groups
    )

    print("\nNa sites in supercell:")

    for name, group in supercell_na_groups.items():
        print(
            f"  {name}: "
            f"{len(group['indices'])} sites"
        )

    configuration_results = calculate_supercell_configurations(
        supercell_na_groups,
        target_percentages
    )

    total_configurations = 1

    for name, result in configuration_results.items():
        total_configurations *= (
            result["number_of_configurations"]
        )

        print(f"\n{name}")
        print(
            f"  Total sites          : "
            f"{result['total_sites']}"
        )
        print(
            f"  Target occupation   : "
            f"{target_percentages[name]:.4f}%"
        )
        print(
            f"  Occupied sites       : "
            f"{result['occupied']}"
        )
        print(
            f"  Vacant sites         : "
            f"{result['vacant']}"
        )
        print(
            f"  Actual occupation    : "
            f"{result['actual_percent']:.4f}%"
        )
        print(
            f"  Configurations       : "
            f"{result['number_of_configurations']:,}"
        )

    print(
        f"\nConfigurations to generate: "
        f"{total_configurations:,}"
    )

    if total_configurations > 100000:
        print(
            "\nWARNING:"
            f"\nThis will generate {total_configurations:,} CIF files."
        )

        confirmation = input(
            "\nType GENERATE to continue: "
        ).strip()

        if confirmation != "GENERATE":
            print("\nGeneration cancelled.")
            return

    base_name = os.path.splitext(
        os.path.basename(cif_file)
    )[0]

    output_directory = Path(
        f"{base_name}_n{selected_n}_"
        f"{supercell_label}_integer_configs"
    )

    generate_configurations(
        supercell=supercell,
        na_groups=supercell_na_groups,
        configuration_results=configuration_results,
        output_directory=output_directory
    )


if __name__ == "__main__":
    main()
