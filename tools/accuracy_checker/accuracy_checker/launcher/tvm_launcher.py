import os
from collections import OrderedDict

from ..config import NumberField, StringField, BoolField
from .launcher import Launcher

import numpy as np
import importlib


class TVMLauncher(Launcher):
    __provider__ = "tvm"

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                "batch": NumberField(
                    value_type=int,
                    min_value=1,
                    optional=True,
                    description="Batch size.",
                    default=1,
                ),
                "device": StringField(
                    choices=["cpu"],
                    optional=True,
                    description="Choose cpu run.",
                    default="cpu",
                ),
                "vm": BoolField(description="Run with Virtual Machine.", default=False),
                "opt_level": NumberField(
                    value_type=int,
                    min_value=0,
                    optional=True,
                    description="Model optimization level",
                    default=2,
                ),
            }
        )
        return parameters

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        try:
            import tvm  # pylint: disable=C0415
            self._tvm = tvm
        except ImportError as import_error:
            raise ValueError(
                "TVM isn't installed. Please, install it before using. \n{}".format(
                    import_error.msg
                )
            )
        self.validate_config(config_entry)

        self._device = self.get_value_from_config("device")
        self._vm_executor = self.get_value_from_config("vm")
        self._batch = self.get_value_from_config("batch")
        self._opt_level = self.get_value_from_config("opt_level")

        self._get_device()

        self._generate_inputs()

        self._module = self._load_module(config_entry["model"])

        self._generate_outputs()

    def _get_device(self):
        if self._device == "cpu":
            self._device = self._tvm.cpu(0)

    def _generate_inputs(self):
        config_inputs = self.config.get("inputs")
        input_shapes = OrderedDict()
        for input_description in config_inputs:
            input_shapes[input_description["name"]] = input_description.get(
                "shape", (self.batch,) + (-1,) * 3
            )
        self._inputs = input_shapes

    def _load_module(self, model_path):
        model_name = os.path.split(model_path)[-1]


        if self.get_value_from_config("vm"):
            print("VirtualMachine Runtime not supported yet. Graph Executor used")
            
        if str(model_path).endswith('tar') == True:
            
            lib = self._tvm.runtime.load_module(model_path)

            graph_executor = importlib.import_module('tvm.contrib.graph_executor')

            return graph_executor.GraphModule(lib["default"](self._device))
            
        elif str(model_path).endswith('json') == True:
            
            params_path = str(model_path).replace('.json', '.params')

            with open(model_path, "r") as file:
                graph_json = file.read()
            with open(params_path, 'rb') as fo:
                params = self._tvm.relay.load_param_dict(fo.read())

            mod = self._tvm.ir.load_json(graph_json)

            with self._tvm.transform.PassContext(opt_level=self._opt_level):
                lib = self._tvm.relay.build(mod, target='llvm', params=params)

            return self._tvm.contrib.graph_executor.GraphModule(lib['default'](self._device))

        else:
            print("Only JSON/params and .tar model supported")
            raise ValueError("Only JSON/params model supported")

    def _generate_outputs(self):
        if self._vm_executor:
            self._outputs_names = self.config.get("outputs")
        else:
            self._outputs_names = list(range(self._module.get_num_outputs()))

    @property
    def inputs(self):
        return self._inputs

    @property
    def batch(self):
        return self._batch

    @property
    def output_blob(self):

        res = 0
        return next(iter(self._outputs_names))

    def fit_to_input(self, data, layer_name, layout, precision):
        data = super().fit_to_input(data, layer_name, layout, precision)
        return self._tvm.nd.array(data, self._device)

    def predict(self, inputs, metadata=None, **kwargs):
        results = []
        input_name = self.config.get("inputs")[0]["name"]
        for batch_input in inputs:
            if self._vm_executor:
                self._module.set_input("main", batch_input[input_name])
                self._module.invoke_stateful("main")
                output = self._module.get_outputs()
                results.append(
                    {
                        output_name: output_value.asnumpy()
                        for output_name, output_value in zip(
                            self._outputs_names, output
                        )
                    }
                )
            else:
                for input_name, input_data in batch_input.items():
                    self._module.set_input(input_name, input_data)


                self._module.run()


                if self.config.get("adapter") == "bert_question_answering":
                    results.append(
                        {
                            str(output_name): self._module.get_output(output_name).asnumpy()
                            for output_name in self._outputs_names
                        }
                    )
                else:
                    results.append(
                        {
                            output_name: self._module.get_output(output_name).asnumpy()
                            for output_name in self._outputs_names
                        }
                    )       
        return results

    def predict_async(self, *args, **kwargs):
        raise ValueError("TVM Launcher does not support async mode yet")

    def release(self):
        del self._module
        del self._device
        del self._vm_executor
        del self._batch
