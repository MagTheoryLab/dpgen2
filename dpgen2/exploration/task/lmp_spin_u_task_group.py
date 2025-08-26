import itertools
import random
import tempfile
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
)

import numpy as np

from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    lmp_traj_name,
    model_name_pattern,
    plm_input_name,
    plm_output_name,
)

from .conf_sampling_task_group import (
    ConfSamplingTaskGroup,
)
from .lmp import (
    make_lmp_input,
)
import dpdata
from .task import (
    ExplorationTask,
)


class LmpSpinUTaskGroup(ConfSamplingTaskGroup):
    def __init__(
        self,
    ):
        super().__init__()
        self.lmp_set = False
        self.plm_set = False
        self.type_map=None
    def set_lmp(
        self,
        numb_models: int,
        lmp_template_fname: str,
        plm_template_fname: Optional[str] = None,
        revisions: dict = {},
        hubbard_u: dict  = {},
    ) -> None:
        if hubbard_u is None or hubbard_u == {}:
            raise ValueError("hubbard_u is required")
        self.hubbard_u = hubbard_u
        self.lmp_template = Path(lmp_template_fname).read_text().split("\n")
        self.revisions = revisions
        self.lmp_set = True
        self.model_list = sorted([model_name_pattern % ii for ii in range(numb_models)])
        if plm_template_fname is not None:
            self.plm_template = Path(plm_template_fname).read_text().split("\n")
            self.plm_set = True

    def make_task(
        self,
    ) -> "LmpSpinUTaskGroup":
        if not self.conf_set:
            raise RuntimeError("confs are not set")
        if not self.lmp_set:
            raise RuntimeError("Lammps SPIN template and revisions are not set")
        # clear all existing tasks
        self.clear()
        confs = self._sample_confs()
        templates = [self.lmp_template]
        for cc  in confs:
            self.revisions["V_APARAM"] = [np.random.choice(self.make_u(cc))]
            conts = self.make_cont(templates, self.revisions)
            nconts = len(conts[0])
            for   ii in   range(nconts ):  # type: ignore
                self.add_task(self._make_lmp_task(cc, conts[0][ii]))
        return self
    def make_u(self,conf:str):
        u_dict= {}
        with tempfile.NamedTemporaryFile() as ft:
            tf = Path(ft.name)
            tf.write_text(conf)
            system = dpdata.System(tf,fmt="lammps/lmp",type_map=self.type_map)
            # print(system.data)
            #首先 先确定目前的conf中有多少体系，然后将u都取出来

            atom_names = np.array(system["atom_names"],dtype=str)
            use_elements = atom_names[np.unique(system["atom_types"])]
            # print(use_elements)
            #['Fe' 'Ge']
            for key in use_elements:
                #这一步可以进一步指定策略
                if key not in self.hubbard_u:
                    continue
                if len(self.hubbard_u[key]["u"])==0:
                    continue
                u_dict[key] = self.hubbard_u[key]["u"]
            keys = list(u_dict.keys())
            values = [u_dict[k] for k in keys]

            combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
            #[{'Fe': 0, 'Ge': 11, 'Nb': 11}]
            # all_atom_names = atom_names[ system["atom_types"] ]
            result = []
            for comb in combinations:
                result.append(" ".join([str(comb.get(elem,0))  for elem in atom_names]))

            return result
    def make_cont(
        self,
        templates: list,
        revisions: dict,
    ):
        keys = revisions.keys()
        prod_vv = [revisions[kk] for kk in keys]
        ntemplate = len(templates)
        ret = [[] for ii in range(ntemplate)]
        for vv in itertools.product(*prod_vv):
            for ii in range(ntemplate):
                tt = templates[ii].copy()
                ret[ii].append("\n".join(revise_by_keys(tt, keys, vv)))
        return ret

    def _make_lmp_task(
        self,
        conf: str,
        lmp_cont: str,
        plm_cont: Optional[str] = None,
    ) -> ExplorationTask:
        task = ExplorationTask()
        task.add_file(
            lmp_conf_name,
            conf,
        ).add_file(
            lmp_input_name,
            lmp_cont,
        )
        if plm_cont is not None:
            task.add_file(
                plm_input_name,
                plm_cont,
            )
        return task



def find_only_one_key(lmp_lines, key):
    found = []
    for idx in range(len(lmp_lines)):
        words = lmp_lines[idx].split()
        nkey = len(key)
        if len(words) >= nkey and words[:nkey] == key:
            found.append(idx)
    if len(found) > 1:
        raise RuntimeError("found %d keywords %s" % (len(found), key))
    if len(found) == 0:
        raise RuntimeError("failed to find keyword %s" % (key))
    return found[0]


def revise_by_keys(lmp_lines, keys, values):
    for kk, vv in zip(keys, values):  # type: ignore
        for ii in range(len(lmp_lines)):
            lmp_lines[ii] = lmp_lines[ii].replace(kk, str(vv))
    return lmp_lines
