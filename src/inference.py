import os
import sys
import logging as log

import numpy as np
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self, model_name, device="CPU", extensions=None):
        self.device = device
        self._init_model(model_name, device, extensions)
        self._check_model(self.core, self.model, device)
        self._init_input_output(self.model)
    
    def _init_model(self, model_name, device, extensions):
        model_weights = model_name+'.bin'
        model_structure = model_name+'.xml'
        self.core = IECore()
        if extensions and "CPU" in device:
            self.core.add_extension(extensions, device)
        try:
            self.model = self.core.read_network(model_structure, model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

    def _check_model(self, core, model, device):
        if "CPU" in device:
            supported_layers = core.query_network(network=model, device_name=device)
            unsupported_layers = [l for l in model.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                print("Unsupported layers found: {}".format(unsupported_layers))
                print("Check whether extensions are available to add to IECore.")
                exit(1)
    
    def _init_input_output(self, model):
        self.input_name = next(iter(model.inputs))
        self.input_shape = model.inputs[self.input_name].shape
        self.output_name = next(iter(model.outputs))
        self.output_shape = model.outputs[self.output_name].shape

    def load_model(self):
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def get_input_shape(self, input_name=None):
        if input_name is None:
            return self.input_shape
        return self.model.inputs[input_name].shape


    def exec_net(self, request_id, inputs):
        if isinstance(inputs, dict):
            self.net.start_async(request_id=request_id, inputs=inputs)
        else:
            self.net.start_async(request_id=request_id, inputs={self.input_name: inputs})
    
    # @multimethod(int, )
    # def exec_net(self)

    def wait(self, request_id):
        status = self.net.requests[request_id].wait()
        return status

    def get_outputs(self, request_id):
        outputs = self.net.requests[request_id].outputs
        return outputs
    
    def get_output(self, request_id):
        output = self.net.requests[request_id].outputs[self.output_name]
        return output