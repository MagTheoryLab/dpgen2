from pathlib import (
    Path,
)
from typing import (
    List, Optional, Dict,
)

import dpdata
import numpy as np
from dargs import (
    Argument,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
)

try:
    from fpop.abacus import (
        AbacusInputs,
        PrepAbacus,
        RunAbacus,
    )
except ModuleNotFoundError:
    AbacusInputs = PrepAbacus = RunAbacus = object

from ..constants import (
    fp_default_out_data_name,
)


class FpOpAbacusInputs(AbacusInputs):  # type: ignore
    @staticmethod
    def args():
        doc_input_file = "A template INPUT file."
        doc_pp_files = (
            "The pseudopotential files for the elements. "
            'For example: {"H": "/path/to/H.upf", "O": "/path/to/O.upf"}.'
        )
        doc_element_mass = (
            "Specify the mass of some elements. "
            'For example: {"H": 1.0079, "O": 15.9994}.'
        )
        doc_kpt_file = "The KPT file, by default None."
        doc_orb_files = (
            "The numerical orbital fiels for the elements, "
            "by default None. "
            'For example: {"H": "/path/to/H.orb", "O": "/path/to/O.orb"}.'
        )
        doc_deepks_descriptor = "The deepks descriptor file, by default None."
        doc_deepks_model = "The deepks model file, by default None."
        return [
            Argument("input_file", str, optional=False, doc=doc_input_file),
            Argument("pp_files", dict, optional=False, doc=doc_pp_files),
            Argument(
                "element_mass", dict, optional=True, default=None, doc=doc_element_mass
            ),
            Argument("kpt_file", str, optional=True, default=None, doc=doc_kpt_file),
            Argument("orb_files", dict, optional=True, default=None, doc=doc_orb_files),
            Argument(
                "deepks_descriptor",
                str,
                optional=True,
                default=None,
                doc=doc_deepks_descriptor,
            ),
            Argument(
                "deepks_model", str, optional=True, default=None, doc=doc_deepks_model
            ),
        ]

class PrepUAbacus(PrepAbacus):
    def prep_task(
            self,
            conf_frame,
            inputs: AbacusInputs,
            prepare_image_config: Optional[Dict] = None,
            optional_input: Optional[Dict] = None,
            optional_artifact: Optional[Dict] = None,
    ):
        r"""Define how one Abacus task is prepared.

        Parameters
        ----------
        conf_frame : dpdata.System
            One frame of configuration in the dpdata format.
        inputs: AbacusInputs
            The AbacusInputs object handels all other input files of the task.
        prepare_image_config: Dict
            Definition of runtime parameters in the process of preparing tasks.
        optional_input:
            Other parameters the developers or users may need.
        optional_artifact
            Other files that users or developers need.
        """


        element_list = conf_frame['atom_names']
        pp, orb = inputs.write_pporb(element_list)
        dpks = inputs.write_deepks()
        mass = inputs.get_mass(element_list)

        if len(orb)==0:
            orb=None

        # if conf_frame has the spins, then we will use it as the initial mag and write it to STRU
        mag = conf_frame.data.get("spins", None)
        if mag is not None:
            mag = mag[0]  # spins is the mag of several frames, here we only use the first frame

        # if the constrain_elements is set, we will set the related flag in STRU
        sc = None
        c_eles = inputs.get_constrain_elements()
        if c_eles:
            atom_names = conf_frame.data["atom_names"]
            atom_types = [atom_names[i] for i in conf_frame.data["atom_types"]]
            sc = [None if i not in c_eles else [1, 1, 1] for i in atom_types]

        conf_frame.to('abacus/stru', 'STRU', pp_file=pp, numerical_orbital=orb, numerical_descriptor=dpks, mass=mass,
                      mag=mag, sc=sc)
        #debug
        # conf_frame.data["hubbard_u"]=np.array([[[1],[1]]])
        if "hubbard_u" in conf_frame.data:
            unique, idx = np.unique(conf_frame.data["atom_types"], return_index=True)
            hubbard_u = conf_frame["hubbard_u"].flatten()
            hubbard_u = hubbard_u[idx]

            inputs.set_input("hubbard_u"," ".join([str(u) for u in hubbard_u]))
            inputs.set_input("dft_plus_u","1")
            atom_names = [conf_frame["atom_names"][i] for i in unique]

            orbital_corr=optional_input.get("orbital_corr", {})
            orbital = [str(orbital_corr.get(elem,2))  if hubbard_u[i] !=0 else   str(-1) for i,elem in enumerate(atom_names)]

            inputs.set_input("orbital_corr"," ".join(orbital))


        inputs.write_input("INPUT")
        inputs.write_kpt("KPT")
        if optional_artifact:
            for file_name, file_path in optional_artifact.items():
                content = file_path.read_text()
                Path(file_name).write_text(content)



class PrepFpOpAbacus(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "type_map": List[str],
                "confs": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_names": BigParameter(List[str]),
                "task_paths": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        confs = []
        # remove atom types with 0 atom from type map, for abacus need pp_files
        # for all atom types in the type map
        for p in ip["confs"]:
            for f in p.rglob("type.raw"):
                system = f.parent
                s = dpdata.System(system, fmt="deepmd/npy")
                atom_numbs = []
                atom_names = []
                for numb, name in zip(s["atom_numbs"], s["atom_names"]):  # type: ignore https://github.com/microsoft/pyright/issues/5620
                    if numb > 0:
                        atom_numbs.append(numb)
                        atom_names.append(name)
                if atom_names != s["atom_names"]:
                    for i, t in enumerate(s["atom_types"]):  # type: ignore https://github.com/microsoft/pyright/issues/5620
                        s["atom_types"][i] = atom_names.index(s["atom_names"][t])  # type: ignore https://github.com/microsoft/pyright/issues/5620
                    s.data["atom_numbs"] = atom_numbs
                    s.data["atom_names"] = atom_names
                    target = "output/%s" % system
                    s.to("deepmd/npy", target)
                    confs.append(Path(target))
                else:
                    confs.append(system)
        op_in = OPIO(
            {
                "inputs": ip["config"]["inputs"],
                "type_map": ip["type_map"],
                "confs": confs,
                "prep_image_config": ip["config"].get("prep", {}),
                "optional_input": ip["config"].get("optional_input", {}),
            }
        )
        op = PrepUAbacus()
        return op.execute(op_in)  # type: ignore in the case of not importing fpop


from typing import (
    Tuple,
)


def get_suffix_calculation(INPUT: List[str]) -> Tuple[str, str]:
    suffix = "ABACUS"
    calculation = "scf"
    for iline in INPUT:
        sline = iline.split("#")[0].split()
        if len(sline) >= 2 and sline[0].lower() == "suffix":
            suffix = sline[1].strip()
        elif len(sline) >= 2 and sline[0].lower() == "calculation":
            calculation = sline[1].strip()
    return suffix, calculation


class RunFpOpAbacus(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_name": BigParameter(str),
                "task_path": Artifact(Path),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "log": Artifact(Path),
                "labeled_data": Artifact(Path),
                "extra_outputs": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        run_config = ip["config"].get("run", {})
        op_in = OPIO(
            {
                "task_name": ip["task_name"],
                "task_path": ip["task_path"],
                "backward_list": [],
                "run_image_config": run_config,
            }
        )
        op = RunAbacus()
        op_out = op.execute(op_in)  # type: ignore in the case of not importing fpop
        workdir = op_out["backward_dir"].parent

        # convert the output to deepmd/npy format
        with open("%s/INPUT" % workdir, "r") as f:
            INPUT = f.readlines()
        _, calculation = get_suffix_calculation(INPUT)
        if calculation == "scf":
            sys = dpdata.LabeledSystem(str(workdir), fmt="abacus/scf")
        elif calculation == "md":
            sys = dpdata.LabeledSystem(str(workdir), fmt="abacus/md")
        elif calculation in ["relax", "cell-relax"]:
            sys = dpdata.LabeledSystem(str(workdir), fmt="abacus/relax")
        else:
            raise ValueError("Type of calculation %s not supported" % calculation)
        out_name = fp_default_out_data_name
        sys.to("deepmd/npy", workdir / out_name)

        extra_outputs = []
        for fname in ip["config"]["extra_output_files"]:
            extra_outputs += list(workdir.glob(fname))

        return OPIO(
            {
                "log": workdir / "log",
                "labeled_data": workdir / out_name,
                "extra_outputs": extra_outputs,
            }
        )

    @staticmethod
    def args():
        doc_cmd = "The command of abacus"
        return [
            Argument("command", str, optional=True, default="abacus", doc=doc_cmd),
        ]
