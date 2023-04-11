# Auto Accuracy Checker

This directory contains `autoac.py` file that automate certain model-related tasks
based on configuration files in the models' directories.

`Auto Accuracy Checker` is a tool that allows you to compile models for the necessary target device using both Graph Executor and Virtual Machine, compile with tuning statistics, generate a new template of accuracy-check.yml configuration file for Accuracy Checker (based on the data from model.yml configuration file) for a specific model or update the existing configuration file with new data obtained and/or changed during the script.

At the moment, this tool is not considered a finished product and its functionality will be supplemented over time.
## Prerequisites

1. Install Python (version 3.6 or higher)
2. Clone this branch to obtain latest actual `tvm_launcher.py` (with the Remote Session and Virtual Machine features)
and `autoac.py`: `git clone https://github.com/Deelvin/open_model_zoo.git -b dbarinov/tvm_launcher_remote_session` OR `git clone git@github.com:Deelvin/open_model_zoo.git -b dbarinov/tvm_launcher_remote_session`
3. <Optional> Before you run the script, you must prepare a directory with
the datasets required for the conversion of annotations (if corresponding flag has been provided, more details [here](#command-line-arguments)) and run of Accuracy Checker itself.
The directory should be located in the same folder as `autoac.py` script, so the path should be structured like this:

```
.
└── open_model_zoo
    ├── ...
    ├── ...
    └── tools
        ├── accuracy_checker
        └── downloader
            ├── datasets  <--
            └── autoac.py
```

You can find more detailed information about dataset preparation in the [Dataset Preparation Guide](https://github.com/Deelvin/open_model_zoo/blob/master/data/datasets.md).

Besides that, you should be able to convert your annotations. 
Read more about that in the [Annotation Converters](https://github.com/Deelvin/open_model_zoo/blob/master/tools/accuracy_checker/accuracy_checker/annotation_converters/README.md).

4. Install the tools and frameworks dependencies with the following command:

```sh
python3 -mpip install --user -r ./requirements.in
```

For models from ONNX:

```sh
pip install onnx==1.9.0
```

For models from MXNet:

```sh
pip install mxnet==1.9.1
```

For models from Keras:

```sh
pip install keras==2.10.0
```

For models from PyTorch:

```sh
python3 -mpip install --user -r ./requirements-pytorch.in
```

For models from TensorFlow:

```sh
python3 -mpip install --user -r ./requirements-tensorflow.in
```
## Command line arguments

These command line arguments are designed to control the behavior of the script depending on your intent:

    '--name', type=str  # Name of the model you intend to download from open_model_zoo
    '--input_model', type=str  # Path to the model (See <this> if you want to use custom model. Otherwise, please, be sure to have at least model.yml with all needed fields in the open_model_zoo/models/public/<model-name> folder)
    '--input', type=str  # Name of the models input (Not necessary if the parameter is present in model.yml file)
    '--outputs', type=str  # Name of the models output/outputs. If model have more than one output name, then provide it like this: --outputs conv2d_58/BiasAdd,conv2d_66/BiasAdd,conv2d_74/BiasAdd without spaces, separated by commas (example yolo-v3-tf model) (Not necessary if the parameter is present in model.yml file)
    '--layout', type=str, choices=['NCHW', 'NHWC'], default="NCHW"  # NCHW or NHWC layout for the model. If --reverse_input_channels in model.yml -> hard set to NHWC.
    '--target', type=str, default='llvm'
    '--target_host', type=str, default='llvm'
    '--precision', type=str, choices=['float16','float16_acc32','float32'], default='float32'  # Accuracy Checker is not stable with a precision different than FP32. However you can --validate your model on any of given precision (only for task_type='classification', dataset='imagenet' at the moment)
    '--session', type=str, choices=['local', 'remote'], default='local'  # Using local session or Remote Session to run (Be sure to connect to your device and export TVM_TRACKER_HOST=x.x.x.x and export TVM_TRACKER_PORT=xxxx environment variables as host/port of tracker)
    '--device', type=str, choices=['cpu', 'gpu'], default='cpu'  # This parameter set dev = tvm.cpu(0) if 'cpu' or dev = tvm.cl(0) if 'gpu'.
    '--vm', type=bool, default=False  # Compile with Virtual Machine (True) or with GraphExecutor (False)
    '--adapter', type=str, default='classification'  # adapter field for accuracy-check.yml (Further plans: pre- and postprocessing depending on the adapter)
    '--model_cfg', type=str  # Path to model.yml if it is not in open_model_zoo/models/public/<model_name> directory
    '--convert_annotations', type=bool, default=False  # Convert annotations for required dataset (dataset field in model.yml -> form cmd_prefix -> run open_model_zoo/tools/accuracy_checker/convert_dataset.py <cmd_prefix> -> obtain converted annotations in the dataset folder)
    '--ac_cfg_path', type=str  # Path to accuracy-check.yml if it is not in open_model_zoo/models/public/<model_name> directory
    '--mode', type=str, choices=['tweak', 'create'], default='default'  # 'default' mode: don't change accuracy-check.yml, just run with it; 'tweak' mode: change only those fields, that were changed during AutoAC script execution; 'create' mode: generate clean .yml file and fill it only with those fields, that were obtained during AutoAC script execution.
    '--log', type=str, default=None  # Path to .log with tuning statistics for the model
    '--validate', type=bool, default=False  # Run validation on 1 sample, obtain accuracy and performance metrics (Only "classification" domain on "cat.png" from "ImageNet" dataset at the moment).

In the [next section](#preparation-for-the-first-launch) we will provide these arguments to make our first launch and in the [AutoAC usage](#autoac-usage) section we will consider some of the most common use-cases of AutoAC.
## Preparation for the first launch
First of all, model configuration file **model.yml** should be prepared. This file consists of following fields:
```yaml
description: is a description of a model
task_type: 'detection' # OR 'classification' OR ... (domain of the network)
adapter: 'yolo_v3' # OR 'classification' OR 'ssd_onnx' OR ... (custom field)
dataset: ms_coco_detection_80_class_without_background  # (dataset name from dataset_definitions.yml)
files: #  (files to be downloaded with their sizes, sha256 hash and link to the source)
  - name: yolo-v3.pb 
    size: 248128731
    sha256: 8ed66d597a936924e98102a5fa16b38569452e17dbb52ddf2877e0f550f57952
    source: https://download.01.org/opencv/public_models/022020/yolo_v3/yolov3.pb
  - name: yolo-v3.json
    size: 384
    sha256: 90f3ea735a2a8908b66dab744b9ec0425fd2564f7b9be6ebd1564dd38d28ec5c
    source: https://download.01.org/opencv/public_models/022020/yolo_v3/yolo_v3_new.json
model_optimizer_args:
  - --reverse_input_channels  # NCHW or NHWC, this case - NHWC
  - --input_shape=[1,3,1200,1200]  # input shape of the model
  - --input=image # input name
  - --mean_values=[123.675,116.28,103.53]  # mean values (normalization)
  - --scale_values=[58.395,57.12,57.375]  # std values (normalization)
framework: tf  # framework
license: https://raw.githubusercontent.com/david8862/keras-YOLOv3-model-set/master/LICENSE # license file
```
Make sure that model.yml file contains all of the fields above, otherwise you should specify it manually.

In order to be able to add your custom model to the OMZ for your personal tests, you need to fill all the fields above with the corresponding information, including links to the source, size in bytes, sha256 hash (See sha256sum command), license file, etc...

Secondly, you should either have fully prepared accuracy-check.yml file (then you can specify path via --ac_cfg_path to it OR put it in models/public/model name folder), or you can create your own configuration file with the basic information about the model and its inference via Accuracy Checker, such as model name, device, session, GraphExecutor or VM, input name, input shape, layout, output names, input precision (not working correctly atm.), adapter, dataset name, preprocessing type (only normalization for now), mean values, scale values, inforamtion about metrics, path to converted annotations (if --convert_annotations True), path to val samples dir (if --convert_annotations True), path to converted dataset meta (if --convert_annotations True)
## AutoAC usage

The basic usage is to run the script like this:

```python
python autoac.py --name <model_name>
```

Make sure that you are located in open_model_zoo/tools/downloader folder. All the downloaded via script models would be located in /public folder.

For better understanding of how the script works, please pay attention to the example use-cases below.

Use Case 1 - Download model from OMZ, read its model.yml and obtain information about model, compile it with obtained information, create base accuracy-check.yml and attempt to run AC.

```
python autoac.py --name <model_name> --input (if not in model.yml) --outputs (if not in model.yml) --layout (usually not required) --target <llvm> --target_host <llvm> --session (if not local) --device (if not cpu) --vm (if not GraphExecutor) --adapter (if not classification) --convert_annotations (if not converted, only for Imagenet and COCO atm.) --mode create
```

Use Case 2 - Specify path to the model, read model.yml on given path and obtain information about model, compile it with obtained information, tweak accuracy-check.yml with obtained model info and attempt to run AC.

```
python autoac.py --input_model <path_to_folder OR path_to_.so_file> --input (if not in model.yml) --outputs (if not in model.yml) --layout (usually not required) --target <llvm> --target_host <llvm> --session (if not local) --device (if not cpu) --vm (if not GraphExecutor) --adapter (if not classification) --convert_annotations (if not converted, only for Imagenet and COCO atm.) --model_cfg (if not in omz/models/public OR custom) --ac_cfg_path (if not in omz/models/public OR custom OR create new on path with --mode create argument) --convert_annotations (if not converted, only for Imagenet and COCO atm.) --mode tweak
```

Use Case 3 - Create folder with your custom model in open_model_zoo/models/public, create model.yml and fill it with required data, compile it with obtained from model.yml data, create accuracy-check.yml manually and run AC on --mode default.

```
python autoac.py --input_model <path_to_folder OR path_to_.so_file> --input (if not in model.yml) --outputs (if not in model.yml) --layout (usually not required) --target <llvm> --target_host <llvm> --session (if not local) --device (if not cpu) --vm (if not GraphExecutor) --adapter (if not classification) --convert_annotations (if not converted, only for Imagenet and COCO atm.) --model_cfg (if not in omz/models/public OR custom) --ac_cfg_path (if not in omz/models/public OR custom OR create new on path with --mode create argument) --convert_annotations (if not converted, only for Imagenet and COCO atm.)
```