import faulthandler
faulthandler.enable()
import argparse
from importlib.resources import path
import sys
import os
import fnmatch
import subprocess

import yaml
from tvm import relay
from tvm.contrib import ndk
from tvm.driver.tvmc.frontends import load_model
#import downloader
import numpy as np
import onnx
from tvm import autotvm

import common


import tvm
from  tvm.relay.op  import  register_mixed_precision_conversion

conv2d_acc = "float32"

@register_mixed_precision_conversion("nn.conv2d", level=11)
def conv2d_mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
    global  conv2d_acc
    return [
        relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
        conv2d_acc,
        mixed_precision_type,
    ] 

@register_mixed_precision_conversion("nn.dense", level=11)
def conv2d_mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
    global  conv2d_acc
    return [
        relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
        conv2d_acc,
        mixed_precision_type,
    ]

def convert_to_dtype(mod, dtype):
    # downcast to float16
    if  dtype == "float16"  or  dtype == "float16_acc32":
        global  conv2d_acc
        conv2d_acc = "float16"  if  dtype == "float16"  else  "float32"
        from  tvm.ir  import  IRModule
        mod = IRModule.from_expr(mod)
        seq = tvm.transform.Sequential(
            [
                relay.transform.InferType(),
                relay.transform.ToMixedPrecision()
            ]
        )
        with  tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
    return  mod


class MyDumper(yaml.Dumper):
    
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

def add_common_args(parser):
    common_args = parser.add_argument_group('Common arguments')
    common_args.add_argument('--name', type=str)
    common_args.add_argument('--input_model', type=str)
    common_args.add_argument('--input', type=str)
    common_args.add_argument('--outputs', type=str)
    common_args.add_argument('--layout', type=str, default="NCHW")
    common_args.add_argument('--target', type=str, default='llvm')
    common_args.add_argument('--target_host', type=str, default='llvm')
    common_args.add_argument('--precision', type=str, choices=['float16','float16_acc32','float32'], default='float32')
    common_args.add_argument('--session', type=str, default='local')
    common_args.add_argument('--device', type=str, default='cpu')
    common_args.add_argument('--vm', type=bool, default=False)
    common_args.add_argument('--adapter', type=str, default='classification')
    common_args.add_argument('--model_cfg', type=str)
    common_args.add_argument('--convert_annotations', type=bool, default=False)
    common_args.add_argument('--ac_cfg_path', type=str)
    common_args.add_argument('--mode', type=str, choices=['tweak', 'create'], default='default')
    common_args.add_argument('--log', type=str, default=None)
    common_args.add_argument('--validate', type=bool, default=False)


def build_arguments_parser():
    parser = argparse.ArgumentParser(description='Deep Learning accuracy validation framework', allow_abbrev=False)
    add_common_args(parser)

    return parser

def get_graphdef_from_tf1(model_path):
    graph_def = None
    tf_model_file = model_path

    import tensorflow as tf
    try:
        tf_compat_v1 = tf.compat.v1
    except ImportError:
        tf_compat_v1 = tf
    # Tensorflow utility functions
    import tvm.relay.testing.tf as tf_testing

    with tf_compat_v1.gfile.GFile(tf_model_file, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    return graph_def

def get_onnx_from_tf1(model_path, model_name, shape_dict, output_names, shape_override = None):
    tf_model_file = model_path
    model_path = os.path.split(model_path)[0]
    input_names = list(shape_dict.keys())

    onnx_model_file = model_path + "/" + model_name + ".onnx"
    if os.path.exists(onnx_model_file) == False:
        import tf2onnx
        import tensorflow as tf
        try:
            tf_compat_v1 = tf.compat.v1
        except ImportError:
            tf_compat_v1 = tf
        # Tensorflow utility functions
        import tvm.relay.testing.tf as tf_testing

        with tf_compat_v1.gfile.GFile(tf_model_file, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            #graph = tf.import_graph_def(graph_def, name="")
            # Call the utility to import the graph definition into default graph.
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)

            model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(graph_def,
                name=model_name, input_names=input_names, output_names=output_names,
                shape_override = shape_override,
                output_path=onnx_model_file)

    return get_mod_params(model_path, shape_dict, output_names, framework='onnx')

def get_dataset_info(dataset_definitions_path, dataset_name):
    with open(dataset_definitions_path) as fh:
        dataset_definitions = yaml.load(fh, Loader=yaml.FullLoader)
        
    for i in range(0, len(dataset_definitions['datasets'])):
        if dataset_definitions['datasets'][i]['name'] == dataset_name:
            dataset_info = dataset_definitions['datasets'][i]
            break
    
    converter = dataset_info['annotation_conversion']['converter']
    try:
        has_background = dataset_info['annotation_conversion']['has_background']
    except KeyError:
        has_background = False
    try:
        sort_annotations = dataset_info['annotation_conversion']['sort_annotations']
    except KeyError:
        sort_annotations = ""
    try:
        use_full_label_map = dataset_info['annotation_conversion']['use_full_label_map']
    except KeyError:
        use_full_label_map = ""
    
    try:
        metrics = dataset_info['metrics']
    except KeyError:
        print("Missing metrics in dataset definition. Provide it manually.")
        metrics = None
    
    return converter, has_background, sort_annotations, use_full_label_map, metrics
        
def get_conversion_string(converter, has_background, sort_annotations, use_full_label_map):
    conversion_string = converter
    
    conversion_string += " --has_background " + str(has_background)
    
    if sort_annotations != '':
        conversion_string += " --sort_annotations " + str(sort_annotations)
    if use_full_label_map != '':
        conversion_string += " --use_full_label_map " + str(use_full_label_map)
    
    return conversion_string
    
def preprocess_config(yaml_path, layout):
    with open(yaml_path) as fh:
        read_data = yaml.load(fh, Loader=yaml.FullLoader)
    
    input_shape, input_name, output_name, framework, dataset_name, mean_values, scale_values = '', '', '', '', '', '', ''
    
    dataset_name = read_data['dataset']
    framework = read_data['framework']
    input_data = read_data['model_optimizer_args']
    shape_pattern = '--input_shape='
    input_pattern = '--input='
    output_pattern = '--output='
    layout_pattern = '--reverse_input_channels'
    mean_values_pattern = '--mean_values='
    scale_values_pattern = '--scale_values='
    if shape_pattern or input_pattern in input_data:
        for index, elem in enumerate(input_data):
            if (elem.find(layout_pattern)!= -1):
                layout = "NHWC"
            if (elem.find(shape_pattern)!= -1):
                string_shape = input_data[index][len(shape_pattern):]
                input_shape = get_shape_from_string(string_shape, 'int')
                print("Input shape: {}".format(input_shape))
            if (elem.find(input_pattern)!= -1):
                input_name = input_data[index][len(input_pattern):]
            if (elem.find(output_pattern)!= -1):
                output_name = input_data[index][len(output_pattern):]
            if (elem.find(mean_values_pattern)!= -1):
                string_mean_values = input_data[index]
                start = string_mean_values.find("[")
                end = string_mean_values.find("]")
                string_mean_values = string_mean_values[start+1:end]
                mean_values = get_shape_from_string(string_mean_values, 'float')
                print("Mean values: {}".format(mean_values))
            if (elem.find(scale_values_pattern)!= -1):
                string_scale_values = input_data[index]
                start = string_scale_values.find("[")
                end = string_scale_values.find("]")
                string_scale_values = string_scale_values[start+1:end]
                scale_values = get_shape_from_string(string_scale_values, 'float')
                print("Scale values: {}".format(scale_values))

    
    return input_shape, input_name, output_name, framework, layout, dataset_name, mean_values, scale_values
    
    
def postprocess_config(config_path, mode, model_name, export_path, device, session, vm, input, shape_dict, layout, outputs, input_precision, adapter, dataset_name, preprocessing_type, mean_values, scale_values, metrics, CONV_PICKLE, VAL_DIR, CONV_META):
    if mode == 'default':
        return
    if mode == 'tweak':
        print("Automatically tweaking accuracy checker config")
        with open(config_path) as fh:
            ac_config = yaml.load(fh, Loader=yaml.FullLoader)
        
        ac_config['models'][0]['name'] = model_name if model_name else ac_config['models'][0].get("name") or 'not_provided'
        ac_config['models'][0]['launchers'][0]['device'] = device if device else ac_config['models'][0]['launchers'][0].get("device") or 'cpu'
        ac_config['models'][0]['launchers'][0]['session'] = session if session else ac_config['models'][0]['launchers'][0].get("session") or 'local'
        ac_config['models'][0]['launchers'][0]['vm'] = vm if vm else ac_config['models'][0]['launchers'][0].get("vm") or False
        
        ac_config['models'][0]['launchers'][0]['inputs'][0]['name'] = input if input else ac_config['models'][0]['launchers'][0]['inputs'][0].get('name') or 'input'
        ac_config['models'][0]['launchers'][0]['inputs'][0]['shape'] = list(shape_dict[input]) if list(shape_dict[input]) else ac_config['models'][0]['launchers'][0]['inputs'][0].get('shape') or 'not_provided'
        ac_config['models'][0]['launchers'][0]['inputs'][0]['layout'] = layout if layout else ac_config['models'][0]['launchers'][0]['inputs'][0].get('layout') or 'NCHW'
        ac_config['models'][0]['launchers'][0]['adapter'] = adapter if adapter else ac_config['models'][0]['launchers'][0].get('adapter') or 'classification'
        ac_config['models'][0]['launchers'][0]['outputs'] = outputs if outputs else ac_config['models'][0]['launchers'][0].get('outputs') or 0
        ac_config['models'][0]['launchers'][0]['_input_precision'] = [input_precision] if input_precision else ac_config['models'][0]['launchers'][0].get('_input_precision') or 'FP32'
        
        ac_config['models'][0]['launchers'][0]['model'] = export_path if export_path else ac_config['models'][0]['launchers'][0].get('model') or 'not_provided'
        
        ac_config['models'][0]['datasets'][0]['name'] = dataset_name if dataset_name else ac_config['models'][0]['datasets'][0].get('name') or 'not_provided'
        ac_config['models'][0]['datasets'][0]['metrics'] = metrics if metrics else ac_config['models'][0]['datasets'][0].get('metrics') or 'not_provided'
        ac_config['models'][0]['datasets'][0]['annotation'] = CONV_PICKLE if CONV_PICKLE else ac_config['models'][0]['datasets'][0].get('annotation') or 'not_provided'
        ac_config['models'][0]['datasets'][0]['data_source'] = VAL_DIR if VAL_DIR else ac_config['models'][0]['datasets'][0].get('data_source') or 'not_provided'
        ac_config['models'][0]['datasets'][0]['dataset_meta'] = CONV_META if CONV_META else ac_config['models'][0]['datasets'][0].get('dataset_meta') or 'not_provided'

        
        for i in range(len(ac_config['models'][0]['datasets'][0]['preprocessing'])):
            if ac_config['models'][0]['datasets'][0]['preprocessing'][i]['type'] == preprocessing_type:
                idx = i + 1
        
        if idx:
            ac_config['models'][0]['datasets'][0]['preprocessing'][idx - 1]['mean'] = mean_values if mean_values else ac_config['models'][0]['datasets'][0]['preprocessing'][idx - 1]['mean']
            ac_config['models'][0]['datasets'][0]['preprocessing'][idx - 1]['std'] = scale_values if scale_values else ac_config['models'][0]['datasets'][0]['preprocessing'][idx - 1]['std']
        else:
            ac_config['models'][0]['datasets'][0]['preprocessing'][len(ac_config['models'][0]['preprocessing'])+1]['type'] = preprocessing_type if preprocessing_type else ac_config['models'][0]['datasets'][0]['preprocessing'][len(ac_config['models'][0]['preprocessing'])+1]['type']
            ac_config['models'][0]['datasets'][0]['preprocessing'][len(ac_config['models'][0]['preprocessing'])+1]['mean'] = mean_values if mean_values else ac_config['models'][0]['datasets'][0]['preprocessing'][len(ac_config['models'][0]['preprocessing'])+1]['mean']
            ac_config['models'][0]['datasets'][0]['preprocessing'][len(ac_config['models'][0]['preprocessing'])+1]['std'] = scale_values if scale_values else ac_config['models'][0]['datasets'][0]['preprocessing'][len(ac_config['models'][0]['preprocessing'])+1]['std']
                
        
    
        with open(config_path, 'w') as fh:
            yaml.dump(ac_config, fh, Dumper=MyDumper, default_flow_style=False, allow_unicode=False, sort_keys=False)
    
    if mode == 'create': 
        print("Creating accuracy checker config")

        yaml_dict = {'models': [{'name': model_name, 'launchers': [{'framework': 'tvm', 'model': export_path, 'device': device, 'session': session, 'vm': vm, 'inputs': [{'name': input, 'type': 'INPUT', 'shape': list(shape_dict[input]), 'layout': layout}], 'outputs': outputs, '_input_precision': [input_precision], 'adapter': {'type': adapter}}], 'datasets': [{'name': dataset_name, 'preprocessing': [{'type': preprocessing_type, 'mean': mean_values, 'std': scale_values}], 'postprocessing': [{'type': '"Here should be postprocessing"'}], 'metrics': metrics, 'annotation': CONV_PICKLE, 'data_source': VAL_DIR, 'dataset_meta': CONV_META}]}]}

        with open(config_path, 'w') as fh:
            yaml.dump(yaml_dict, fh, Dumper=MyDumper, default_flow_style=False, allow_unicode=False, sort_keys=False)

def get_model_info(args):
    return [model for model in common.load_models(args)
            if fnmatch.fnmatchcase(model.name, args.name)][0]


def get_shape_from_string(string, dtype):
    processed = string.replace(' ', '')
    processed = processed.replace('[', '')
    processed = processed.replace(']', '')
    processed = processed.split(',')
    if dtype == 'int':
        return list(int(entry) for entry in processed)
    elif dtype == 'float':
        return list(float(entry) for entry in processed)


def get_mod_params(model_path, shape_dict, outputs, framework):
    model_name = os.path.split(model_path)[-1]
    input = list(shape_dict.keys())[0]
    
    if framework == 'onnx':
        # .onnx
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(".onnx"):
                    path_to_onnx = f'{root}/{file}'
                    model = onnx.load(path_to_onnx)
                    mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)
                    mod = relay.transform.DynamicToStatic()(mod)
        return mod, params
    elif framework == 'mxnet':
        import mxnet as mx
        import mxnet.gluon as gluon
        # .params and .json
        names_dict = {'mxnet-vgg16': 'vgg16'}
        if model_name in names_dict.keys():
            mxnet_name = names_dict[model_name]
            model = gluon.model_zoo.vision.get_model(mxnet_name, pretrained=True)
            mod, params = relay.frontend.from_mxnet(model, shape_dict)
            return mod, params
        else:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith(".params"):
                        path_to_params = f'{root}/{file}'
                    if file.endswith(".json"):
                        path_to_json = f'{root}/{file}'
        
            net2 = mx.gluon.SymbolBlock.imports(path_to_json, [input], path_to_params)
            mod, params = relay.frontend.from_mxnet(net2, shape_dict)
        return mod, params
    elif framework == 'pytorch':
        import torch
        import torchvision
        names_dict = {
            "resnet-50-pytorch": "resnet50",
            "resnet-18-pytorch": "resnet18",
            "resnet-34-pytorch": "resnet34",
            "mobilenet-v2-pytorch": "mobilenetv2",
            "shufflenet-v2-x1.0": "shufflenet_v2_x1_0",
        }
        if model_name in names_dict.keys():
            pytorch_name = names_dict[model_name]
            model = getattr(torchvision.models, pytorch_name)(pretrained=True)
        model = model.eval()
        input_shape = shape_dict[input]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        shape_list = [(input, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        return mod, params
    elif framework == 'tf':
        # if .h5 -> keras if .pb -> graph_def
        import tensorflow as tf
        import keras
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(".h5"):
                    path_to_h5 = f'{root}/{file}'
                    names_dict = {
                        "densenet-121-tf": "DenseNet121",
                        "densenet-169-tf": "DenseNet169",
                        "densenet-201-tf": "DenseNet201"
                    }
                    if model_name in names_dict.keys():
                        keras_name = names_dict[model_name]
                        tf.keras.backend.set_image_data_format('channels_last')
                        model = getattr(tf.keras.applications, keras_name)(weights=path_to_h5)
                        path_to_model = '/'.join(str(path_to_h5).split('/')[:-1])
                        savedmodel_output = f'{path_to_model}/savedmodel'
                        model.save(filepath=savedmodel_output)
                        os.system(f"python -m tf2onnx.convert --saved-model {savedmodel_output} --output {path_to_model}/{model_name}.onnx")
                    path_to_onnx = f'{path_to_model}/{model_name}.onnx'
                    return get_mod_params(path_to_onnx, shape_dict, outputs, "onnx")
                if file.endswith(".pb"):
                    path_to_pb = f'{root}/{file}'
                    return get_onnx_from_tf1(path_to_pb, model_name, shape_dict, outputs)
                    # graph_def = get_graphdef_from_tf1(path_to_pb)
                    # mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict,
                    #                     outputs=outputs)
                    # return mod, params
                    
                    
    return "Unknown framework"

def main(cli_parser: argparse.ArgumentParser):
    argv = cli_parser.parse_args()
    print("Argv: {}".format(argv))
    import tvm
    from pathlib import Path
    
    OMZ_ROOT = Path(__file__).resolve().parents[2] 
    DATASET_DEFINITIONS = OMZ_ROOT / 'tools' / 'accuracy_checker' / 'dataset_definitions.yml'
    MODELS_ROOT = OMZ_ROOT / 'models' / 'public'
    DOWNLOADED_MODELS_ROOT = OMZ_ROOT / 'tools' / 'downloader' / 'public'
    AC_PATH = OMZ_ROOT / 'tools' / 'accuracy_checker' / 'accuracy_check.py'
    
    if not argv.input_model:
        try:
            model_path = DOWNLOADED_MODELS_ROOT / argv.name
            import downloader
            old_args = sys.argv
            sys.argv = ['downloader.py', '--name', argv.name]
            downloader.main()
            sys.argv = old_args
        except NameError as ne:
            print("Model {} not found in OMZ. Please, provide path to model with --input_model".format(ne))
    else:
        model_path = argv.input_model
        if not os.path.isdir(model_path):
            model_path = '/'.join(str(model_path).split('/')[:-1])
        if not argv.name:
            argv.name = os.path.split(model_path)[-1]
    
    MODEL_CFG = MODELS_ROOT / argv.name / 'model.yml'
    AC_CFG = MODELS_ROOT / argv.name / 'accuracy-check.yml'
    
    if not argv.model_cfg:
        if not MODEL_CFG.exists():
            raise FileNotFoundError(f"Model config not found in root folder: {MODEL_CFG}. Please specify it manually.")
        argv.model_cfg = MODEL_CFG
    
    if not argv.ac_cfg_path:
        if not AC_CFG.exists():
            raise FileNotFoundError(f"AC config not found in root folder: {AC_CFG}. Please specify it manually or select new creation of .yml via providing '--mode create' flag and path to new cfg '--ac_cfg_path path/to/.yml'.")
        argv.ac_cfg_path = AC_CFG
    
    if not argv.layout:
        argv.layout = 'NCHW'
    
    input_shape, input_name, output_name, framework, layout, dataset_name, mean_values, scale_values = preprocess_config(argv.model_cfg, argv.layout)
    
    shape_dict = {input_name: input_shape}
    
    argv.input = input_name if not argv.input else argv.input
    argv.layout = layout if not argv.layout else argv.layout
    argv.outputs = output_name if not argv.outputs else argv.outputs
    outputs = argv.outputs.split(',') if argv.outputs else None
    
    preprocessing_type = None
    if mean_values or scale_values:
        preprocessing_type = "normalization"
        
    if argv.validate:
        if argv.adapter == 'classification':
            from PIL import Image
            from tvm.contrib.download import download_testdata
            import numpy as np

            img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
            img_path = download_testdata(img_url, "cat.png", module="data")
            img = Image.open(img_path)
            img = img.resize(shape_dict[argv.input][1:-1])
            img = np.array(img) - np.array([127.5, 127.5, 127.5])
            img /= np.array([127.5, 127.5, 127.5])
            #img = img.transpose((2, 0, 1))
            img = img[np.newaxis, :]
    
    
    try:
        from google.protobuf.message import DecodeError
        mod, params = get_mod_params(model_path, shape_dict, outputs, framework)
    except DecodeError:
        print("DecodeError: Error parsing message with type 'onnx.ModelProto'. Try to delete the model file(-s) and download it again.")
        return 1
    
    converter, has_background, sort_annotations, use_full_label_map, metrics = get_dataset_info(DATASET_DEFINITIONS, dataset_name)
    
    if argv.convert_annotations:
        print('Converting annotations...')
        cmd_prefix = get_conversion_string(converter, has_background, sort_annotations, use_full_label_map)
        print("Prefix for conversion: {}".format(cmd_prefix))
        if "imagenet" in cmd_prefix:
            src_path = "Imagenet/Imagenet"
            annotations_path = f"{src_path}/ILSVRC2012_val.txt"
            pickle = "imagenet.pickle"
            if "--has_background True" in cmd_prefix:
                json = "imagenet.json"
        elif "mscoco_detection" in cmd_prefix:
            src_path = "COCO/COCO"
            annotations_path = f"{src_path}/annotations/instances_val2017_small10.json"
            src_path = "COCO/COCO/val2017"
            pickle = "mscoco_detection.pickle"
            if "--has_background True" in cmd_prefix:
                json = "mscoco_detection.json"
        
        DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", )
        VAL_DIR = os.path.join(DATASET_DIR, src_path)
        print("Datasets root folder: {}".format(DATASET_DIR))
        ANNOTATIONS = os.path.join(DATASET_DIR, annotations_path)
        print("Annotations path: {}".format(ANNOTATIONS))
        CONV_ANNOTATIONS_DIR = f"{DATASET_DIR}/annotations/"
        print("Converted annotations root folder: {}".format(CONV_ANNOTATIONS_DIR))
        CONV_PICKLE = f"{CONV_ANNOTATIONS_DIR}{pickle}"
        print("Pickle file: {}".format(CONV_PICKLE))
        if json:
            CONV_META = f"{CONV_ANNOTATIONS_DIR}{json}"
            print("Dataset had metadata. Path to META: {}".format(CONV_META))
        else:
            CONV_META = ""
        os.system(f"python /Users/admin/workspace/open_model_zoo/tools/accuracy_checker/convert_dataset.py {cmd_prefix} --annotation_file {ANNOTATIONS} --output_dir {CONV_ANNOTATIONS_DIR}")
        print('Annotations converted')
    else:
        VAL_DIR = ''
        CONV_PICKLE = '' 
        CONV_META = ''
        
    from tvm.contrib import graph_executor
    import tvm.runtime.vm
    import numpy as np
    
    target = tvm.target.Target(argv.target, host=argv.target_host)
 
    dtype = argv.precision
    if not dtype == 'float32':
        mod = convert_to_dtype(mod["main"], dtype)
    _input_precision = f"{argv.input}:FP32"

    print("IR representation of the model:\n{}".format(mod))
    
    if argv.vm:
        print("Compiling with VM")
        import copy
        from tvm.relay import vm
        if argv.log:
            print("Apply tuning statistics: {}".format(argv.log))
            with autotvm.apply_history_best(argv.log):
                with tvm.transform.PassContext(opt_level=3):
                    graph_module = vm.compile(copy.deepcopy(mod), target=target, target_host=target.host, params=params)
            export_path = os.path.join(model_path, f"exec_{argv.name}_{argv.device}_{argv.session}_vm_{dtype}_tuned.so")
        else:
            with tvm.transform.PassContext(opt_level=3):
                graph_module = vm.compile(copy.deepcopy(mod), target=target, target_host=target.host, params=params)
            export_path = os.path.join(model_path, f"exec_{argv.name}_{argv.device}_{argv.session}_vm_{dtype}.so")
        lib = graph_module.mod
    else:
        print("Compiling with Graph Executor")
        if argv.log:
            print("Apply tuning statistics: {}".format(argv.log))
            with autotvm.apply_history_best(argv.log):
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build(mod, target=target, target_host=target.host, params=params)
            export_path = os.path.join(model_path, f"{argv.name}_{argv.device}_{argv.session}_{dtype}_tuned.so")
        else:
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, target_host=target.host, params=params)
            export_path = os.path.join(model_path, f"{argv.name}_{argv.device}_{argv.session}_{dtype}.so")
    
    dtype = "float32"  if  dtype == "float32"  else  "float16"
    
    if argv.session == "remote":
        import tvm.rpc
        rpc_tracker_host = os.environ.get("TVM_TRACKER_HOST")
        rpc_tracker_port = int(os.environ.get("TVM_TRACKER_PORT"))
        key="android"

        print("Connecting to remote device...")
        tracker = tvm.rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port)
        remote = tracker.request(key, priority=0, session_timeout=600000)
        if argv.device == "gpu":
            print("Running on GPU on remote device")
            dev = remote.cl(0)
        else:
            print("Running on CPU on remote device")
            dev = remote.cpu(0)
        print("Connected to remote device")
        lib.export_library(export_path, ndk.create_shared)
        print("Exported library to: ", export_path)
        model_name = os.path.split(export_path)[-1]
        remote.upload(export_path)
        print("Lib has been uploaded to target device {}.".format(dev))
        lib = remote.load_module(model_name)
        print("Loaded library on remote device")
    else:
        if argv.device == "gpu":
            print("Running on GPU on local device") 
            dev = tvm.cl(0)
        else:
            print("Running on CPU on local device")
            dev = tvm.cpu(0)
        lib.export_library(export_path)

    
    if argv.vm:
        m = tvm.runtime.vm.VirtualMachine(lib, dev)
        if argv.validate:
            m.set_input("main", tvm.nd.array(img.astype("float32"), device=dev))
        else:
            m.set_input("main", tvm.nd.array(np.zeros(shape_dict[argv.input], dtype="float32"), device=dev))
        m.invoke_stateful("main")
        tvm_output = m.get_outputs()
    else:
        m = graph_executor.GraphModule(lib["default"](dev))
        if argv.validate:
            m.set_input(argv.input, tvm.nd.array(img.astype("float32")))
        else:
            m.set_input(argv.input, tvm.nd.array(np.random.normal(size=shape_dict[argv.input]).astype("float32")))
        m.run()
        tvm_output = m.get_output(0)

    ######################################################################
    # Get predictions and performance statistic
    # -----------------------------------------
    # This piece of code displays the top-1 and top-5 predictions, as
    # well as provides information about the model's performance
    
    if argv.validate:
        if argv.adapter == 'classification':
            from  os.path  import  join, isfile
            from  matplotlib  import  pyplot  as  plt
            from  tvm.contrib  import  download

            # Download ImageNet categories
            categ_url = "https://github.com/uwsampl/web-data/raw/main/vta/models/"
            categ_fn = "synset.txt"
            download.download(join(categ_url, categ_fn), categ_fn)
            synset = eval(open(categ_fn).read())

            top_categories = np.argsort(tvm_output.asnumpy()[0])
            top5 = np.flip(top_categories, axis=0)[:5]

            # Report top-1 classification result
            print("Top-1 id: {}, class name: {}".format(top5[1-1], synset[top5[1-1]]))

            # Report top-5 classification results
            print("\nTop5 predictions: \n")
            print("\t#1:", synset[top5[1-1]])
            print("\t#2:", synset[top5[2-1]])
            print("\t#3:", synset[top5[3-1]])
            print("\t#4:", synset[top5[4-1]])
            print("\t#5:", synset[top5[5-1]])
            print("\t", top5)
            ImageNetClassifier = False
            for  k  in  top_categories[-5:]:
                if  "cat"  in  synset[k]:
                    ImageNetClassifier = True
            assert  ImageNetClassifier, "Failed ImageNet classifier validation check"

            print("Evaluate inference time cost...")
            print(m.benchmark(dev, number=1, repeat=10))
    
    
    
    
    
    
    postprocess_config(argv.ac_cfg_path, argv.mode, argv.name, export_path, argv.device, argv.session, argv.vm, argv.input, shape_dict, argv.layout, argv.outputs, _input_precision, argv.adapter, dataset_name, preprocessing_type, mean_values, scale_values, metrics, CONV_PICKLE, VAL_DIR, CONV_META)
    # go to .bashrc (or .zshrc) and add the following line: export ACCURACY_CHECKER_DIR=/Users/admin/workspace/open_model_zoo/tools/accuracy_checker 
    # and add it to PYTHONPATH variable so it looks like this -> export PYTHONPATH=$TVM_HOME/python:$ACCURACY_CHECKER_DIR:${PYTHONPATH}
    # write 'source .bashrc' (or source .zshrc) in terminal to apply changes
    # at this point you can keep only 'main(build_arguments_parser())' in 'if __name__ == '__main__':' and remove everything else.
    # If everything is done correctly, you should be able to run the script and get the accuracy check results (Uncomment all below that line):
    # import accuracy_check as ac
    # old_args = sys.argv
    # sys.argv = ['accuracy-check.py', '-c', argv.ac_cfg_path]
    # ac.main()
    # sys.argv = old_args

    return AC_PATH, argv.ac_cfg_path

if __name__ == '__main__':
    accuracy_checker_path, accuracy_config_path = main(build_arguments_parser())
    p = subprocess.Popen(['python', accuracy_checker_path, '-c', accuracy_config_path])
    p.wait()
    p.terminate()
    sys.exit(0)
