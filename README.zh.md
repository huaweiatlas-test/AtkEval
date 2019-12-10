[EN](README.en.md)|CN

## 模型精度对比工具

本文介绍模型精度对比工具。该工具适用于tensorflow及caffe模型，运行于Atlas 300产品。当tensorflow或caffe模型成功转换为适用于 Ascend 310 芯片的模型（以下简称DaVinci模型）时，可通过本工具实现对模型转换前后推理精度的对比。本工具支持三种不同的对比模式：（1）逐层对比模式，逐算子对比tensorflow或caffe模型与DaVinci模型的推理结果差异；（2）二分对比模式，通过二分网络并选取精度差异较大的分支进行对比，最终得到差异最大的算子所在，该模式尤其适用于规模较大的模型（3）特定对比模式，由用户指定模型的中部分连续子网结构，仅对比该子网的输出差异。用户可根据需求选择上述任一方式运行本工具。本文主要介绍运行本工具所需的环境配置和具体步骤。

[TOC]

### 支持的产品

 Atlas 300 (Model 3010)

### 支持的版本

1.31.T15.B150 或 1.3.2.B893

可通过执行以下命令获取

```
npu-smi info
```

### 运行环境

Ubuntu 16.04 或 CentOS 7.0
Python >= 3.5.2 (Python 2 is not supported)
tensorflow >= 1.12
caffe == 1.0

### 目录结构

本工具的目录结构如下：

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

其中user.config为用户配置文件，main.py为本工具主脚本；caffe_impl和tf_impl分别为caffe和tensorflow的功能模块的实现；utils中为caffe和tensorflow均需使用的公共接口，目录/utils/davinci_infer下为DaVinci模型推理的可执行文件和C++源代码，编译方法参考下一节。

### 使用方法

#### 1.编译DaVinci推理模块

本工具包含DaVinci推理模块，该部分为C++代码，需编译。请按如下步骤进行编译。

（1）进入DaVinci推理模块目录

```
cd utils/davinci_infer/
```

（2）添加环境变量DDK_PATH，请使用系统中实际的DDK目录

```
export DDK_PATH=/home/Atlas/tools/che/ddk/ddk
```

（3）执行build.sh编译

```
./build.sh
```

该模块由本工具自动调用。

#### 2.配置user.config

在每次运行模型精度对比工具前，需配置user.config文件。user.config使用python3模块configparser解析，其格式可参考链接 https://docs.python.org/3.5/library/configparser.html#supported-ini-file-structure。user.config中的各字段说明如下：

| section     | key    | 说明                                                         |
| ----------- | ------ | ------------------------------------------------------------ |
| framework   | path   | 原始模型使用的框架，仅支持“tensorflow”或“caffe”两个名称      |
| ddk         | path   | ddk路径                                                      |
| caffe       | path   | pycaffe路径，若仅运行tensorflow模型，该字段内容忽略          |
| model       | path   | 模型文件路径，tensorflow模型为pb文件的路径，caffe模型为prototxt文件路径 |
| weight      | path   | caffe权重文件caffemodel路径，仅caffe模型需要                 |
| mode        | number | 运行模式，仅支持“1”、“2”、“3”三种模式，其中1为逐层对比，2为特定对比，3为二分对比 |
| start_nodes | name   | 仅特定对比模式需要填写，子网的输入节点名称，若有多个节点，以“，”号隔开，例如"resnet50/input1, resnet50/input2" |
| end_nodes   | name   | 仅特定对比模式需要填写，子网的输出节点名称，若有多个节点，以“，”号隔开，例如"resnet50/output1, resnet50/outputput2" |
| result      | path   | 对比结果文件保存的路径                                       |

用户需正确配置本文件以运行本工具。下面举例说明了两次不同的配置。

配置1：对一个tensorflow模型进行特定对比

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

配置2：对一个caffe模型进行逐层对比

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

#### 3.执行

由于Atlas 300安全限制，执行本工具的Linux用户必须属于HwHiAiUser组。运行前，请确保当前用户已切换到HwHiAiUser组。

执行精度对比工具命令如下，
$ python3 main.py


执行完成后，会打印类似如下对比结果：

```
Node_name               euc_distance        rel_distance        cos_distance        
resnet_v2_50_node_1     2e-06               0.00169             0.999997            
resnet_v2_50_node_2     1e-06               0.006474            0.999998            
```

其中，Node_name列表示节点名称，euc_distance、 rel_distance、cos_distance分别为该节点在原始tensorflow或caffe中的输出和DaVinci模型中的输出的欧式距离、相对误差、余弦相似度。不同对比模式下打印的内容有一定区别，具体如下表：

| 对比模式          | 打印内容                                 |
| ----------------- | ---------------------------------------- |
| 逐层对比（模式1） | 打印模型中所有算子的对比结果             |
| 特定对比（模式2） | 仅打印模型子网输出算子的对比结果         |
| 二分对比（模式3） | 仅打印模型中误差最大的一个算子的对比结果 |

上述打印内容同样会以文本形式存入usr.config中设定的结果保存路径中。

### 备注

在使用特定对比模式时，用户需确保起始-终止节点组合确定的子网结构是有效且完整的。

```
node0    node1
  |        |
node2    node3
   \      /
     node4
       |
     node5
```

假设原始模型结构如上所示，下面列举了几种无法形成有效子网的起始-终止节点对：

（1）终止节点无法反向连通到起始节点，例如start_nodes:node0;end_nodes:node1

（2）缺少部分起始节点，例如start_nodes:node2;end_nodes:node5

（3）多个起始节点之间存在连通关系，例如start_nodes:node3，node4;end_nodes:node5