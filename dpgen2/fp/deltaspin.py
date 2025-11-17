import logging
from pathlib import (
    Path,
)
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import dpdata
import numpy as np
from dargs import (
    Argument,
    ArgumentEncoder,
    Variant,
    dargs,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    FatalError,
    OPIOSign,
    TransientError,
)

from dpgen2.constants import (
    fp_default_log_name,
    fp_default_out_data_name,
)
from dpgen2.utils.run_command import (
    run_command,
)

from fpop.prep_fp import (
    PrepFp,
)
from .run_fp import (
    RunFp,
)
from .vasp import loads_incar, dumps_incar
from .vasp_input import (
    VaspInputs,
    make_kspacing_kpoints,
)

# global static variables
vasp_conf_name = "POSCAR"
vasp_input_name = "INCAR"
vasp_pot_name = "POTCAR"
vasp_kp_name = "KPOINTS"


class DeltaspinInput(VaspInputs):
    def __init__(self,
                 kspacing: Union[float, List[float]],
                 incar: str,
                 pp_files: Dict[str, str],
                 kgamma: bool = True,
                 constrain_elements: Optional[List[str]] = None,
                 ):
        super().__init__(kspacing,incar,pp_files,kgamma)
        self.constrain_elements=constrain_elements


class PrepDeltaSpin(PrepFp):
    def prep_task(
        self,
        conf_frame,
        inputs: DeltaspinInput,
        prepare_image_config: Optional[Dict] = None,
        optional_input: Optional[Dict] = None,
        optional_artifact: Optional[Dict] = None,

    ):
        r"""Define how one DeltaSpin task is prepared.

        Parameters
        ----------
        conf_frame : dpdata.System
            One frame of configuration in the dpdata format.
        vasp_inputs : VaspInputs
            The VaspInputs object handels all other input files of the task.
        """
        params = loads_incar(inputs.incar_template)

        if "hubbard_u" in conf_frame.data:
            sort_idx = np.argsort(conf_frame.data["atom_types"])
            hubbard_u = conf_frame["hubbard_u"].flatten()
            hubbard_u = hubbard_u[sort_idx]
            unique, idx = np.unique(hubbard_u, return_index=True)
            hubbard_u = unique[np.argsort(idx)]


            atom_names = conf_frame["atom_names"]
            orbital_corr=optional_input.get("orbital_corr", {})
            orbital = [str(orbital_corr.get(elem,2))  if hubbard_u[i] !=0 else   str(-1) for i,elem in enumerate(atom_names)]



            params["LDAU"] = True
            params["LDAUTYPE"] = 2
            params["LDAUL"] = " ".join(orbital)
            params["LDAUU"] = " ".join([str(u) for u in hubbard_u])
            params["LDAUJ"] = " ".join([str(0) for u in hubbard_u])
            params["LDAUPRINT"] = 2
            params["LASPH"] = True

        if "spins"  in conf_frame.data:
            magmom=conf_frame.data["spins"][0]

            magmom = [" ".join([f"{i:.4f}" for i in sublist]) + " \\\n" for sublist in magmom]
            params["MAGMOM"] = " ".join(magmom)
            params["M_CONSTR"] = " ".join(magmom)
            params["LAMBDA"]=f"{conf_frame.get_natoms()*3}*0"
            constrl=""
            for num ,atom in zip(conf_frame.data["atom_numbs"],conf_frame.data["atom_names"]):
                if atom in inputs.constrain_elements:
                    constrl+=f"{num*3}*1 "
                else:
                    constrl+=f"{num*3}*0 "
            params["CONSTRL"]=constrl

        incar = dumps_incar(params)

        Path(vasp_input_name).write_text(incar)
        conf_frame.to("vasp/poscar", vasp_conf_name)
        # fix the case when some element have 0 atom, e.g. H0O2
        tmp_frame = dpdata.System(vasp_conf_name, fmt="vasp/poscar")
        Path(vasp_pot_name).write_text(inputs.make_potcar(tmp_frame["atom_names"]))
        Path(vasp_kp_name).write_text(inputs.make_kpoints(conf_frame["cells"][0]))  # type: ignore



class PrepFpDeltaSpin(OP):
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
        op = PrepDeltaSpin()
        return op.execute(op_in)  # type: ignore in the case of not importing fpop




class RunDeltaSpin(RunFp):
    def input_files(self) -> List[str]:
        r"""The mandatory input files to run a DeltaSpin task.

        Returns
        -------
        files: List[str]
            A list of madatory input files names.

        """
        return [vasp_conf_name, vasp_input_name, vasp_pot_name, vasp_kp_name]

    def optional_input_files(self) -> List[str]:
        r"""The optional input files to run a DeltaSpin task.

        Returns
        -------
        files: List[str]
            A list of optional input files names.

        """
        return []

    def run_task(
        self,
        command: str,

        out: str,
        log: str,
    ) -> Tuple[str, str]:
        r"""Defines how one FP task runs

        Parameters
        ----------
        command : str
            The command of running vasp task
        out : str
            The name of the output data file.
        log : str
            The name of the log file

        Returns
        -------
        out_name: str
            The file name of the output data in the dpdata.LabeledSystem format.
        log_name: str
            The file name of the log.
        """

        log_name = log
        out_name = out
        # run vasp
        command = " ".join([command, ">", log_name])
        ret, out, err = run_command(command, shell=True)
        if ret != 0:
            logging.error(
                "".join(
                    (
                        "DeltaSpin failed\n",
                        "out msg: ",
                        out,
                        "\n",
                        "err msg: ",
                        err,
                        "\n",
                    )
                )
            )
            raise TransientError("DeltaSpin failed")
        # convert the output to deepmd/npy format
        sys = dpdata.LabeledSystem("OUTCAR", fmt="vasp_deltaspin/outcar")
        sys.to("deepmd/npy", out_name )
        return out_name, log_name

    @staticmethod
    def args():
        r"""The argument definition of the `run_task` method.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments of `run_task` method.
        """

        doc_deltaspin_cmd = "The command of DeltaSpin"

        doc_deltaspin_log = "The log file name of DeltaSpin"
        doc_deltaspin_out = "The output dir name of labeled data. In `deepmd/spin/npy` format provided by `dpdata`."
        return [
            Argument(
                "command",
                str,
                optional=True,
                default="vasp_deltaspin",
                doc=doc_deltaspin_cmd,
            ),

            Argument(
                "out",
                str,
                optional=True,
                default=fp_default_out_data_name,
                doc=doc_deltaspin_out,
            ),
            Argument(
                "log",
                str,
                optional=True,
                default=fp_default_log_name,
                doc=doc_deltaspin_log,
            ),
        ]
