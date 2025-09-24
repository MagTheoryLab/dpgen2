import os
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Type,
)

from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Step,
    Steps,
    Workflow,
    argo_len,
    argo_range,
    argo_sequence,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    BigParameter,
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    Parameter,
    PythonOPTemplate,
    Slices,
)


from dpgen2.constants import (
    fp_index_pattern,
)
from dpgen2.utils.step_config import (
    init_executor,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict


class PrepRunFp(Steps):
    def __init__(
        self,
        name: str,
        prep_op: Type[OP],
        run_op: Type[OP],
        prep_config: Optional[dict] = None,
        run_config: Optional[dict] = None,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        prep_config = normalize_step_dict({}) if prep_config is None else prep_config
        run_config = normalize_step_dict({}) if run_config is None else run_config
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
            "fp_config": InputParameter(),
            "type_map": InputParameter(),
        }
        self._input_artifacts = {"confs": InputArtifact()}
        self._output_parameters = {
            "task_names": OutputParameter(),
        }
        self._output_artifacts = {
            "logs": OutputArtifact(),
            "labeled_data": OutputArtifact(),
            "extra_outputs": OutputArtifact(),
        }

        super().__init__(
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )

        self._keys = ["prep-fp", "run-fp"]
        self.step_keys = {}
        ii = "prep-fp"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "run-fp"
        self.step_keys[ii] = "--".join(
            ["%s" % self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )

        self = _prep_run_fp(
            self,
            self.step_keys,
            prep_op,
            run_op,
            prep_config=prep_config,
            run_config=run_config,
            upload_python_packages=upload_python_packages,
        )

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def keys(self):
        return self._keys


def _prep_run_fp(
    prep_run_steps,
    step_keys,
    prep_op: Type[OP],
    run_op: Type[OP],
    prep_config: dict = normalize_step_dict({}),
    run_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    prep_executor = init_executor(prep_config.pop("executor"))
    run_executor = init_executor(run_config.pop("executor"))
    template_slice_config = run_config.pop("template_slice_config", {})

    prep_fp = Step(
        "prep-fp",
        template=PythonOPTemplate(
            prep_op,
            output_artifact_archive={"task_paths": None},
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "config": prep_run_steps.inputs.parameters["fp_config"],
            "type_map": prep_run_steps.inputs.parameters["type_map"],
        },
        artifacts={
            "confs": prep_run_steps.inputs.artifacts["confs"],
        },
        key=step_keys["prep-fp"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_steps.add(prep_fp)

    run_fp = Step(
        "run-fp",
        template=PythonOPTemplate(
            run_op,
            slices=Slices(
                "int('{{item}}')",
                input_parameter=["task_name"],
                input_artifact=["task_path"],
                output_artifact=["log", "labeled_data", "extra_outputs"],
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "task_name": prep_fp.outputs.parameters["task_names"],
            "config": prep_run_steps.inputs.parameters["fp_config"],
        },
        artifacts={
            "task_path": prep_fp.outputs.artifacts["task_paths"],
        },
        with_sequence=argo_sequence(
            argo_len(prep_fp.outputs.parameters["task_names"]), format=fp_index_pattern
        ),
        # with_param=argo_range(argo_len(prep_fp.outputs.parameters["task_names"])),
        key=step_keys["run-fp"],
        executor=run_executor,
        **run_config,
    )

    prep_run_steps.add(run_fp)

    prep_run_steps.outputs.parameters[
        "task_names"
    ].value_from_parameter = prep_fp.outputs.parameters["task_names"]
    prep_run_steps.outputs.artifacts["logs"]._from = run_fp.outputs.artifacts["log"]
    prep_run_steps.outputs.artifacts["labeled_data"]._from = run_fp.outputs.artifacts[
        "labeled_data"
    ]
    prep_run_steps.outputs.artifacts["extra_outputs"]._from = run_fp.outputs.artifacts[
        "extra_outputs"
    ]

    return prep_run_steps

class ComputeFpDispatch(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "task_names": BigParameter(List[str]),
                "dispatch_groups": BigParameter(List[Dict]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "splits": BigParameter(List[Dict[str, int]]),
            }
        )

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        task_count = len(ip["task_names"])
        groups = list(ip["dispatch_groups"])
        if len(groups) == 0:
            return OPIO({"splits": []})

        weights: List[float] = []
        for grp in groups:
            weight = grp.get("weight", 1.0)
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = 1.0
            if weight < 0:
                weight = 0.0
            weights.append(weight)

        weight_sum = sum(weights)
        if weight_sum <= 0:
            weights = [1.0 for _ in weights]
            weight_sum = float(len(weights))

        counts = []
        remainders: List[float] = []
        for weight in weights:
            raw = task_count * weight / weight_sum
            floor = int(raw)
            counts.append(floor)
            remainders.append(raw - floor)

        allocated = sum(counts)
        remaining = task_count - allocated
        if remaining > 0:
            order = sorted(
                range(len(groups)),
                key=lambda idx: (
                    -remainders[idx],
                    -weights[idx],
                    idx,
                ),
            )
            for idx in order[:remaining]:
                counts[idx] += 1
        elif remaining < 0:
            order = sorted(
                range(len(groups)),
                key=lambda idx: (
                    remainders[idx],
                    weights[idx],
                    -idx,
                ),
            )
            for idx in order[: -remaining]:
                if counts[idx] > 0:
                    counts[idx] -= 1

        splits: List[Dict[str, int]] = []
        start = 0
        for idx, cnt in enumerate(counts):
            stop = start + cnt
            splits.append(
                {
                    "index": groups[idx].get("index", idx),
                    "start": start,
                    "stop": stop,
                }
            )
            start = stop

        if splits and splits[-1]["stop"] < task_count:
            splits[-1]["stop"] = task_count
        return OPIO({"splits": splits})


class SelectFpTasks(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "task_names": BigParameter(List[str]),
                "task_paths": Artifact(List[Path]),
                "splits": BigParameter(List[Dict[str, int]]),
                "group_index": Parameter(int),
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
    def execute(self, ip: OPIO) -> OPIO:
        splits = ip["splits"]
        group_index = ip["group_index"]
        if group_index >= len(splits):
            return OPIO({"task_names": [], "task_paths": []})
        split = splits[group_index]
        start = split.get("start", 0)
        stop = split.get("stop", start)
        task_names = ip["task_names"][start:stop]
        task_paths = ip["task_paths"][start:stop]

        return OPIO({"task_names": task_names, "task_paths": task_paths})


class MergeFpOutputs(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {

                "logs_list": Artifact(List[Path]),
                "labeled_data_list": Artifact(List[Path]),
                "extra_outputs_list": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {

                "logs": Artifact(List[Path]),
                "labeled_data": Artifact(List[Path]),
                "extra_outputs": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:


        logs=[log_path   for log_path in ip["logs_list"]  if "dflow_logs_list" not  in log_path.name]
        labeled_data=[labeled_data   for labeled_data in ip["labeled_data_list"]  if "dflow_labeled_data_list" not  in labeled_data.name]
        extra_outputs_list=[extra_output   for extra_output in ip["extra_outputs_list"]  if "dflow_extra_outputs_list" not  in extra_output.name]

        return OPIO(
            {

                "logs":logs,
                "labeled_data": labeled_data,
                "extra_outputs": extra_outputs_list,
            }
        )


class PrepSplitRunFp(Steps):
    def __init__(
        self,
        name: str,
        prep_op: Type[OP],
        run_op: Type[OP],
        run_configs: List[dict],
        dispatch_groups: List[Dict],
        prep_config: Optional[dict] = None,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        assert len(run_configs) > 0, "run_configs should not be empty"
        self._dispatch_groups = dispatch_groups
        prep_config = normalize_step_dict({}) if prep_config is None else prep_config
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
            "fp_config": InputParameter(),
            "type_map": InputParameter(),
            "dispatch_groups": InputParameter(type=list, value=dispatch_groups),
        }
        self._input_artifacts = {"confs": InputArtifact()}
        self._output_parameters = {
            "task_names": OutputParameter(),
        }
        self._output_artifacts = {
            "logs": OutputArtifact(),
            "labeled_data": OutputArtifact(),
            "extra_outputs": OutputArtifact(),
        }

        super().__init__(
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )
        self.inputs.parameters["dispatch_groups"].value = dispatch_groups
        self._keys = ["prep-fp", "plan-dispatch", "select-fp", "run-fp", "merge-fp"]
        self.step_keys = {}
        self.step_keys["prep-fp"] = "--".join([
            "%s" % self.inputs.parameters["block_id"],
            "prep-fp",
        ])

        self = _prep_split_run_fp(
            self,
            prep_op,
            run_op,
            dispatch_groups=dispatch_groups,
            prep_config=prep_config,
            run_configs=run_configs,
            upload_python_packages=upload_python_packages,
        )

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def keys(self):
        return self._keys


def _prep_split_run_fp(
    prep_run_steps,
    prep_op: Type[OP],
    run_op: Type[OP],
    dispatch_groups: List[Dict],
    prep_config: dict = normalize_step_dict({}),
    run_configs: Optional[List[dict]] = None,
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    if run_configs is None:
        run_configs = []
    assert len(run_configs) == len(dispatch_groups), "run configs do not match dispatch groups"

    prep_config = deepcopy(prep_config)
    prep_template_config = prep_config.pop("template_config")
    prep_executor = init_executor(prep_config.pop("executor"))

    prep_fp = Step(
        "prep-fp",
        template=PythonOPTemplate(
            prep_op,
            output_artifact_archive={"task_paths": None},
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "config": prep_run_steps.inputs.parameters["fp_config"],
            "type_map": prep_run_steps.inputs.parameters["type_map"],
        },
        artifacts={
            "confs": prep_run_steps.inputs.artifacts["confs"],
        },
        key=prep_run_steps.step_keys["prep-fp"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_steps.add(prep_fp)

    dispatch_step = Step(
        "plan-dispatch",
        template=PythonOPTemplate(
            ComputeFpDispatch,
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "task_names": prep_fp.outputs.parameters["task_names"],
            "dispatch_groups": prep_run_steps.inputs.parameters["dispatch_groups"],
        },
        artifacts={},
        key="--".join([
            "%s" % prep_run_steps.inputs.parameters["block_id"],
            "plan-dispatch",
        ]),
        executor=prep_executor,
    )
    prep_run_steps.add(dispatch_step)

    select_steps = []
    run_steps = []

    for idx, run_cfg in enumerate(run_configs):

        select_step = Step(
            f"select-fp-{idx}",
            template=PythonOPTemplate(
                SelectFpTasks,
                python_packages=upload_python_packages,
                **prep_template_config,
            ),
            parameters={
                "task_names": prep_fp.outputs.parameters["task_names"],
                "splits": dispatch_step.outputs.parameters["splits"],
                "group_index": idx,
            },
            artifacts={
                "task_paths": prep_fp.outputs.artifacts["task_paths"],
            },
            key="--".join([
                "%s" % prep_run_steps.inputs.parameters["block_id"],
                f"select-fp-{idx}",
            ]),
            executor=prep_executor,
        )

        select_steps.append(select_step)

        run_cfg_copy = deepcopy(run_cfg)
        run_cfg_copy.pop("__meta__", None)

        run_template_config = run_cfg_copy.pop("template_config")
        run_executor = init_executor(run_cfg_copy.pop("executor"))
        template_slice_config = run_cfg_copy.pop("template_slice_config", {})

        run_step = Step(
            f"run-fp-{idx}",
            template=PythonOPTemplate(
                run_op,
                slices=Slices(
                    "int('{{item}}')",
                    input_parameter=["task_name"],
                    input_artifact=["task_path"],
                    output_artifact=["log", "labeled_data", "extra_outputs"],
                    **template_slice_config,
                ),
                python_packages=upload_python_packages,
                **run_template_config,
            ),
            parameters={
                "task_name": select_step.outputs.parameters["task_names"],
                "config": prep_run_steps.inputs.parameters["fp_config"],
            },
            artifacts={
                "task_path": select_step.outputs.artifacts["task_paths"],
            },
            with_sequence=argo_sequence(
                argo_len(select_step.outputs.parameters["task_names"]),
                format=fp_index_pattern,
            ),
            key="--".join([
                "%s" % prep_run_steps.inputs.parameters["block_id"],
                f"run-fp-{idx}" + "-{{item}}",
            ]),
            executor=run_executor,
            **run_cfg_copy,
        )

        run_steps.append(run_step)
    prep_run_steps.add(select_steps)
    prep_run_steps.add(run_steps)
    from dflow.step import  argo_enumerate
    merge_step = Step(
        "merge-fp",
        template=PythonOPTemplate(
            MergeFpOutputs,
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={

        },
        artifacts={
            "logs_list": [run_step.outputs.artifacts["log"] for run_step in run_steps],
            "labeled_data_list": [
                run_step.outputs.artifacts["labeled_data"] for run_step in run_steps
            ],
            "extra_outputs_list": [
                run_step.outputs.artifacts["extra_outputs"] for run_step in run_steps
            ],
        },
        key="--".join([
            "%s" % prep_run_steps.inputs.parameters["block_id"],
            "merge-fp",
        ]),
        executor=prep_executor,
    )

    prep_run_steps.add(merge_step)

    prep_run_steps.outputs.parameters[
        "task_names"
    ].value_from_parameter = prep_fp.outputs.parameters["task_names"]
    prep_run_steps.outputs.artifacts["logs"]._from = merge_step.outputs.artifacts["logs"]
    prep_run_steps.outputs.artifacts["labeled_data"]._from = merge_step.outputs.artifacts[
        "labeled_data"
    ]
    prep_run_steps.outputs.artifacts["extra_outputs"]._from = merge_step.outputs.artifacts[
        "extra_outputs"
    ]

    return prep_run_steps

