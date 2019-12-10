EN|[CN](README.zh.md)

## DaVinci Model Precision Analyzing Tool

This document describes the model precision comparison tool. This tool applies to the TensorFlow and Caffe models and runs on the Atlas 300. When the TensorFlow or Caffe model is successfully converted to the model applicable to the Ascend 310 chip (DaVinci model for short), you can use this tool to compare the inference precision before and after the model conversion.  This tool supports three comparison modes: (1) Layer by layer comparison mode:This mode compares the inference results of the TensorFlow or Caffe model with those of the Da Vinci model by operator. (2) Binary comparison mode:This mode is applicable to large-scale models. In this mode, binary networks are used and branches with great precision differences are selected for comparison to obtain the operator with the largest difference. (3)  Specific comparison mode:In this mode, users specify some continuous subnet structures in the model and compare only the output differences of the subnets. You can run the tool in any of the preceding modes as required.

This document describes how to configure the environment for running the tool.

[TOC]

### Supported Products

Atlas 300 (Model 3010)

### Supported Version

1.31.T15.B150 or 1.3.2.B893

It can be obtained by executing the following command

```
npu-smi info
```

### Compatible Environments

Ubuntu 16.04 or CentOS 7.0
Python >= 3.5.2 (Python 2 is not supported)
tensorflow >= 1.12
caffe == 1.0

### Directory Structure

The directory structure of this tool is shown as follows：

```
├── caffe_impl
│   ├── caffe_bindary_compare.py
│   ├── caffe_compare.py
│   ├── caffe_subnet_extract.py
│   ├── caffe_successive_compare.py
│   └── caffe_util.py
├── __init__.py
├── main.py
├── tf_impl
│   ├── data_save.py
│   ├── __init__.py
│   ├── tf_basic_compare.py
│   ├── tf_binary_split.py
│   └── tf_subnet_extract.py
├── user.config
└── utils
    ├── common_info.py
    ├── convert2davinci.py
    ├── davinci_infer
    ├── davinci_run.py
    ├── __init__.py
    └── precision_compare.py
```

user.config is the user configuration file. main.py is the main script of this tool.The files in the caffe_impl and tf_impl directory are the implementations of the functional modules of the Caffe and TensorFlow respectively. The files in the utils directory are the public interfaces used by both the Caffe and TensorFlow.
The executable file and C++ source code of the Da Vinci model inference are stored in the /utils/davinci_infer. For details about the compilation method, see the next section.

### User Guide to DaVinci Model Precision Analyzing Tool

#### 1.Compilation of DaVinci inference module

This tool includes a DaVinci inference module，which is written with C++ and needs compilation. Please compile it as follows.

(1) Enter in the  directory of DaVinci inference module

```
cd utils/davinci_infer/
```

(2) Add environment variable DDK_PATH, please use the actual DDK directory in your system

```
export DDK_PATH=/home/Atlas/tools/che/ddk/ddk
```

(3) Execute build.sh to compile

```
./build.sh
```

The module is automatically called by the tool.

#### 2.Configure the user.config

Before running the model precision comparison tool, you need to configure the user.config file. The user.config file is parsed by the configparser of the Python3 module. For details about the format, see the link https://docs.python.org/3.5/library/configparser.html#supported-ini-file-structure. The fields in user.config are described as follows:

| section     | key    | description                                                  |
| ----------- | ------ | ------------------------------------------------------------ |
| framework   | name   | The AI framework adopted by the original model，only 'tensorflow' or 'caffe' is supported. |
| ddk         | path   | The path of ddk.                                             |
| pycaffe     | path   | The path of pycaffe, this section will be ignored when running tensorflow model. |
| model       | path   | The path of model file. The path of 'pb' file and 'prototxt' file for tensorflow and caffe, respectively. |
| weight      | path   | The path of 'caffemodel' file, this section will be ignored when running tensorflow model. |
| mode        | number | run mode. Layer by layer comparison mode, dichotomy comparison mode and sub net comparison mode are supported. Digits 1, 2, and 3 are used to represent those modes respectively. |
| start_nodes | name   | In sub net comparison mode, start nodes of the sub net should be given explicitly. If there are multiple start nodes, they should be given separated with ',', e.g. 'resnet50/input1, resnet50/input2'. |
| end_nodes   | name   | In sub net comparison mode, end nodes of the sub net should be given explicitly. If there are multiple end nodes, they should be given separated with ',', e.g. 'resnet50/output1, resnet50/outputput2'. |
| result      | path   | The directory where comparison result will be saved.         |

You need to correctly configure this file to run the tool. The following example shows two different configurations.

Configuration 1：Compare a TensorFlow model in the specific comparison mode.

```python
[ddk]
path = /home/Atlas/tools/che/ddk/ddk
[framework]
name = tensorflow
[model]
path = resnet_v2_50.pb
[mode]
number = 2
[result]
path = result
[start_nodes]
name =resnet_v2_50/block1/unit_1/bottleneck_v2/preact/FusedBatchNorm
[end_nodes]
name =resnet_v2_50/block1/unit_1/bottleneck_v2/shortcut/BiasAdd,   resnet_v2_50/block1/unit_1/bottleneck_v2/conv3/BiasAdd
```

Configuration 2：Compare a Caffe model in the layer-by-layer mode.

```python
[ddk]
path = /home/Atlas/tools/che/ddk/ddk
[caffe]
path = /home/caffe/python
[framework]
name = caffe
[model]
path = resnet_v2_50.prototxt
[weight]
path = resnet_v2_50.caffemodel
[mode]
number = 1
[result]
path = result
```

#### 3.Excution

Due to Atlas 300 security restrictions, the Linux user who runs this tool must belong to the HwHiAiUser group.
Switch to HwHiAiUser group to execute the tools.
$ python3 main.py


After the command is executed, the following comparison result is displayed.

```
Node_name               euc_distance        rel_distance        cos_distance        
resnet_v2_50_node_1     2e-06               0.00169             0.999997            
resnet_v2_50_node_2     1e-06               0.006474            0.999998            
```

The Node_name column indicates the node name. euc_distance, rel_distance, and cos_distance indicate the Euclidean distance, relative error, and cosine similarity between the output of the node in the original TensorFlow or Caffe model and the output of the node in the DaVinci model respectively. The following table lists the differences between the displayed content in different comparison modes.

| Comparison mode                    | Displayed content                                            |
| ---------------------------------- | ------------------------------------------------------------ |
| Layer-by-layer comparison (mode 1) | The comparison results of all operators in the model.        |
| Specific comparison (mode 2)       | Only the comparison results of subnet output operators in the model. |
| Binary comparison (mode 3)         | Only the comparison result of the operator with the maximum error in the model. |

The printed content above will also be saved to the path set in the user.config file in the form of text .

### Remarks

When using the specific comparison mode, users need to ensure that the subnet structure determined by the combination of start and end nodes is effective and complete.

```
node0    node1
  |        |
node2    node3
   \      /
     node4
       |
     node5
```

Assuming that the structure of the original model is as shown above, here are a few start/end node pairs that cannot form an effective subnet:

(1) The end nodes cannot access to the start node in reverse, such as start_nodes:node0;end_nodes:node1

(2) Some start nodes are missing, such as start_nodes:node2;end_nodes:node5

(3) There is connectivity between any two of the start nodes，such as start_nodes:node3，node4;end_nodes:node5