from aiida import load_profile, orm
from aiida.engine import submit
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from pymatgen.core import Structure

load_profile("CSCS")


structure = orm.StructureData(pymatgen=Structure.from_file)"/home/ababio_p/screened_cifs/Na0.494V2O5.cifs/R1A28_0.494.cif"))
code = orm.load_code("pw7.5@localhost")
kpoints = orm.KpointsData()
kpoints.set_kpoints_mesh([4, 2, 4])

#I am not sure if i import the pwrelaxworkchain i am supposed to define
#calculation in CONTROL. Kindly let me know
pw_parameters = {
        "CONTROL": {
            "calculation": "vc-relax",
            "restart_mode": "from_scratch",
            "tstress": True,
            "tprnfor": True,
            "verbosity": high,
            "nstep": 500,
        },
        "SYSTEM": {
            "ecutwfc": 50,
            "ecutrho": 400,
            "nspin": 2,
            "ntyp": 3,
            "nosym": False,
            "starting_magnetization": {"V": 0.1},
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


builder = PwRelaxWorkChain.get_builder_from_protocol(
    code,
    structure,
    pseudo_family="SSSP/1.3/PBEsol/efficiency",
    overrides={
    "base_init_relax": {
        "pw": {
            "parameters": pw_parameters,}_c
    },
    "base_relax": {
        "pw": {
            "parameters": pw_parameters,
        }
    }'
}
)

builder.base_init_relax.kpoints = kpoints
builder.base_relax.kpoints = kpoints
node = sunmit(builder)
print(node.pk)
