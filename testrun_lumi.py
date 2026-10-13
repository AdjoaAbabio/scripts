#!/usr/bin/env python3

from pathlib import Path

from aiida import load_profile, orm
from aiida.engine import submit
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.common.types import RelaxType
from pymatgen.core import Structure



PROFILE = "CSCS"

CODE_LABEL = "pw@lumi-c"

CIF_PATH = Path(
    "/home/ababio_p/screened_cifs/Na0.494V2O5_n2_cifs/R1S28_0.494.cif"
)

PSEUDO_FAMILY = "SSSP/1.3/PBEsol/efficiency"

KPOINTS_MESH = [4, 2, 4]


pw_parameters = {
    "CONTROL": {
        "tstress": True,
        "tprnfor": True,
        "nstep": 500,
    },

    "SYSTEM": {
        "ecutwfc": 50,
        "ecutrho": 400,
        "nspin": 2,
        "nosym": False,
        "starting_magnetization": {
            "V": 0.1,
        },
        "occupations": "smearing",
        "smearing": "mv",
        "degauss": 0.01,
    },

    "ELECTRONS": {
        "conv_thr": 1.2e-8,
        "electron_maxstep": 200,
        "mixing_beta": 0.4,
    },
}


load_profile(PROFILE)

if not CIF_PATH.is_file():
    raise FileNotFoundError(
        f"\nCIF file not found:\n{CIF_PATH}\n"
        "Check the file path before submitting."
    )

pymatgen_structure = Structure.from_file(str(CIF_PATH))

structure = orm.StructureData(
    pymatgen=pymatgen_structure
)

structure.label = CIF_PATH.stem


print("\nSTRUCTURE INFORMATION")
print("----------------------")
print(f"File: {CIF_PATH.name}")
print(f"Formula: {pymatgen_structure.composition.reduced_formula}")
print(f"Number of sites: {len(pymatgen_structure)}")


# load quantum espresso lumi-c

code = orm.load_code(CODE_LABEL)

print("\nCODE INFORMATION")
print("----------------")
print(f"Code label: {code.label}")
print(f"Computer: {code.computer.label}")


if code.computer.label == "localhost":
    raise RuntimeError(
        "\nERROR: The selected Quantum ESPRESSO code points to "
        "localhost rather than LUMI-C.\n"
        "Check the code registration before submitting."
    )


# create pwrelaxworkchain builder

builder = PwRelaxWorkChain.get_builder_from_protocol(
    code=code,
    structure=structure,
    protocol="moderate",
    relax_type=RelaxType.POSITIONS_CELL,
    pseudo_family=PSEUDO_FAMILY,
)


# apply calculation parameters

for stage_name in ("base_init_relax", "base_relax"):

    stage = getattr(builder, stage_name)

    # Obtain the protocol-generated parameters.
    parameters = stage.pw.parameters.get_dict()

    # Merge the user-defined settings into the existing parameters.
    for section, values in pw_parameters.items():

        parameters.setdefault(section, {}).update(values)

    # Update the builder with the modified parameters.
    stage.pw.parameters = orm.Dict(dict=parameters)


# set kpoint mesh

kpoints = orm.KpointsData()

kpoints.set_kpoints_mesh(KPOINTS_MESH)

builder.base_init_relax.kpoints = kpoints
builder.base_relax.kpoints = kpoints


# workflow metadata

builder.metadata.label = (
    f"vc-relax_{CIF_PATH.stem}"
)

builder.metadata.description = (
    "Variable-cell structural relaxation of "
    f"{CIF_PATH.name} using Quantum ESPRESSO on LUMI-C."
)


# Display information

print("\nSUBMISSION INFORMATION")
print("----------------------")
print(f"Profile: {PROFILE}")
print(f"Code: {CODE_LABEL}")
print(f"Computer: {code.computer.label}")
print(f"Pseudopotential family: {PSEUDO_FAMILY}")
print(f"K-point mesh: {KPOINTS_MESH}")
print("Calculation: Variable-cell relaxation")
print("Workflow: PwRelaxWorkChain")


# submit workflow

print("\nSubmitting workflow to AiiDA...")

node = submit(builder)

print("\nSUBMISSION SUCCESSFUL")
print("---------------------")
print(f"Workflow PK: {node.pk}")
print(f"Workflow UUID: {node.uuid}")

print("\nTo inspect the workflow:")
print(f"verdi process show {node.pk}")

print("\nTo monitor running processes:")
print("verdi process list -a")
