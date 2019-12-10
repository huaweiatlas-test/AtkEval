from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.framework.test_ops import _InitOpDefLibrary

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def op_def_lib():
    ''' create op def library for Placeholder op
    '''
    return _InitOpDefLibrary(
        b"\nC\n\013Placeholder\032\017\n\006output\"\005dtype\"\r\n\005dtype"
        b"\022\004type\"\024\n\005shape\022\005shape\032\004:\002\030\001")


def extract_keep_name(start_nodes, end_nodes, name_to_input_name,
                      name_to_seq_num):
    ''' extract nodes to keep in the sub graphdef according to
        start_nodes and end_nodes
    Args:
        start_nodes: a list whose element is start node name
        end_nodes: a list whose element is end node name
        name_to_input_name: a dict to find the input node name of a node
        name_to_seq_num: a dict to find the sequence index of a node
    Return:
        node_name_keep: a list including all the node names to keep in the
        extracted sub graphdef.It should be noted that some ops like Conv2D
        has weight and if those nodes are in start_nodes, then the weight
        nodes are NOT in the node_name_keep.
    '''
    nodes_to_keep = set()
    next_to_visit = end_nodes[:]
    while next_to_visit:
        per_node_name = next_to_visit[0]
        del next_to_visit[0]
        if per_node_name in nodes_to_keep:
            continue
        nodes_to_keep.add(per_node_name)
        if per_node_name not in start_nodes:
            next_to_visit += name_to_input_name[per_node_name]
    node_name_keep = sorted(
        list(nodes_to_keep),
        key=lambda per_node_name: name_to_seq_num[per_node_name])
    return node_name_keep


def nodes_check(name_to_node, name_to_input_name, name_to_seq_num, start_nodes,
                end_nodes):
    ''' check the validation of start_nodes and end_nodes
    Args:
        name_to_node: a dict whose keys are node names and values are nodedef
        start_nodes: a list whose element is start node name
        end_nodes: a list whose element is end node name
    Return:
        if start_nodes and end_nodes are valid, return True,
        otherwise return False
    '''
    if not isinstance(start_nodes, list):
        print('start_nodes must be a list')
        return False
    if not isinstance(end_nodes, list):
        print('end_nodes must be a list')
        return False
    for per_node_name in start_nodes:
        if not isinstance(per_node_name, str):
            print('element of start_nodes must be str')
            return False
        if per_node_name not in name_to_node.keys():
            print("start_node %s is not in graph" % per_node_name)
            return False
    for per_node_name in end_nodes:
        if not isinstance(per_node_name, str):
            print('element of end_nodes must be str')
            return False
        if per_node_name not in name_to_node.keys():
            print("end_nodes %s is not in graph" % per_node_name)
            return False

    # not supported pattern in specific comparing mode: if two of the start
    # nodes are accessed to each other.
    for start_node_first in start_nodes:
        for start_node_second in start_nodes:
            if start_node_first != start_node_second:
                node_name_keep = extract_keep_name([start_node_first],
                                                   [start_node_second],
                                                   name_to_input_name,
                                                   name_to_seq_num)
                if start_node_first in node_name_keep:
                    print('invalid start nodes, any two of the start nodes '
                          'should not be accessed to each other')
                    return False

    return True


def node_name(per_node_name):
    '''return the actual node name'''
    if per_node_name.startswith("^"):
        return per_node_name[1:]

    return per_node_name.split(":")[0]


def input_shape_fix(input_shape):
    '''if the shape is None, give a fixed value'''
    input_shape_fixed = []
    for idx, dim_shape in enumerate(input_shape.as_list()):
        if dim_shape is None:
            if idx == 0:
                input_shape_fixed.append(1)
            else:
                input_shape_fixed.append(64)
        else:
            input_shape_fixed.append(dim_shape)
    input_shape = tf.TensorShape(input_shape_fixed)
    return input_shape


def tf_subnet_extract(graph_def,
                      graph,
                      start_nodes,
                      end_nodes,
                      queue,
                      sub_graph_node_list=None):
    ''' extract the sub graphdef. Two modes are supported:
        1.If sub_graph_node_list is not None, then the sub graph is extracted
          using sub_graph_node_list, the start_nodes are required and end_nodes
          will be ignored. All the inputs of the start_nodes will be replaced
          with Placeholder except some certain types and those nodes in
          sub_graph_node_list.
        2.If sub_graph_node_list is None,then the sub graph is extracted using
         start_nodes and end_nodes,All the inputs of the start_nodes will be
         replaced with Placeholder except some certain types.

        the extracted sub graphdef and the description of of the inputs of the
         start_nodes will be put in the queue and
        returned
    Args:
        graph_def: global graphdef of the model
        graph: global graph imported from graphdef
        start_nodes: a list whose element is start node name
        end_nodes: a list whose element is end node name
        queue: a multiprocessing Queue
        sub_graph_node_list: nodes name list including all nodes in the sub
        graph
    Return:
        all the returned items are put in a multiprocessing Queue
    '''
    if sub_graph_node_list is not None:
        name_to_node = {}
        for node in graph_def.node:
            name_to_node[node_name(node.name)] = node
        node_name_keep = sub_graph_node_list

    else:

        name_to_input_name, name_to_node, name_to_seq_num = \
            _extract_graph_summary(graph_def)

        ret = nodes_check(name_to_node, name_to_input_name,
                          name_to_seq_num, start_nodes, end_nodes)

        if not ret:
            return queue.put([None, None])

        node_name_keep = extract_keep_name(start_nodes, end_nodes,
                                           name_to_input_name,
                                           name_to_seq_num)

        for name in node_name_keep:
            if name_to_node[name].op == 'Placeholder' and \
                    name not in start_nodes:
                print('The specific sub graph is NOT valid, please check!')
                return queue.put([None, None])

    input_description = list()

    name_to_node_backup = copy.deepcopy(name_to_node)

    for per_node_name in start_nodes:
        if name_to_node_backup[per_node_name].input:
            input_shape_alternative = []
            for i in range(len(name_to_node_backup[per_node_name].input)):
                if (sub_graph_node_list is not None) and (
                        name_to_node_backup[per_node_name].input[i] in
                        sub_graph_node_list):
                    continue
                # when one node has more than one output, the second ends with
                # :1, the third ends with :2, and so on
                input_node_name = name_to_node[per_node_name].input[i].split(
                    ':')[0]

                # for conv2d and some other ops which have weights, the weights
                #  are fed as a Const op through
                # Identity op
                if name_to_node[input_node_name].op == 'Identity' and \
                    len(name_to_node[input_node_name].input) == 1 \
                        and name_to_node[name_to_node[input_node_name].input[
                            0]].op == 'Const':
                    node_name_keep.insert(
                        0, name_to_node[per_node_name].input[i])
                    node_name_keep.insert(
                        0, name_to_node[
                            name_to_node[per_node_name].input[i]].input[0])
                    input_shape_alternative = tf.compat.v1.graph_util. \
                        tensor_shape_from_node_def_name(
                            graph, name_to_node[input_node_name].input[0])

                # some ops have params fed as Const op directly
                elif name_to_node[input_node_name].op == 'Const':
                    node_name_keep.insert(
                        0, name_to_node[per_node_name].input[i])

                else:
                    # get the output type of the input of the current node
                    # usually, the output type is identical to the input type
                    # of one node, in this case, 'T' represents
                    # the input and output type of one node. If the ouput type
                    # of one node is different from input type,
                    # we need to use the 'out_type' as the output type of the
                    # input of the current node
                    if 'out_type' in name_to_node[input_node_name].attr.keys():
                        input_node_type = getattr(
                            name_to_node[input_node_name].attr['out_type'],
                            'type')  # get the output type of this node

                    elif 'T' in name_to_node[input_node_name].attr.keys():
                        input_node_type = getattr(
                            name_to_node[input_node_name].attr['T'],
                            'type')  # get the output type of this node

                    else:
                        input_node_type = types_pb2.DT_FLOAT

                    # if one node only has one output, the split len is 1,
                    # otherwise more than 1
                    if len(name_to_node[per_node_name].input[i].split(
                            ':')) == 1:

                        input_shape = tf.compat.v1.graph_util. \
                            tensor_shape_from_node_def_name(
                                graph,
                                name_to_node[per_node_name].input[i])

                        # special case for MatMul with reshape op input
                        if name_to_node[
                                per_node_name].op == 'MatMul' and not any(
                                    input_shape.as_list()):
                            input_shape = input_shape_alternative

                        # when the type of the input is Scalar, the shape is []
                        # davinci model not supports Scalar input ,so we need
                        # to skip those node
                        if not input_shape:
                            return queue.put([None, None])

                        # omg need fixed input shapes rather than 'None' shape
                        # here we roughly fix the 'None' in the first dim to \
                        # 1 and other 'None' to 64
                        input_shape = input_shape_fix(input_shape)

                        name_to_node[per_node_name].input[i] = name_to_node[
                            per_node_name].input[i] + '_placeholder'

                        node_name_keep.insert(
                            0, name_to_node[per_node_name].input[i])

                        # create a Placeholder op with specified dtype, shape
                        # and name
                        _, _, _op = op_def_lib()._apply_op_helper(
                            "Placeholder",
                            dtype=input_node_type,
                            shape=input_shape,
                            name=name_to_node[per_node_name].input[i])
                        name_to_node[name_to_node[per_node_name].
                                     input[i]] = _op.node_def

                        # save the input description of the sub graphdef
                        # including name, shape and type
                        desc = dict()
                        desc['name'] = name_to_node[per_node_name].input[i]
                        desc['shape'] = input_shape
                        desc['dtype'] = input_node_type
                        input_description.append(copy.deepcopy(desc))

                    else:
                        input_shape = tf.compat.v1.graph_util. \
                            tensor_shape_from_node_def_name(
                                graph,
                                name_to_node[per_node_name].input[i])
                        if not input_shape:
                            return queue.put([None, None])

                        input_shape = input_shape_fix(input_shape)

                        input_tensor_name = name_to_node[per_node_name].input[
                            i].split(':')[0] + '_' + name_to_node[
                                per_node_name].input[i].split(':')[1]

                        name_to_node[per_node_name].input[
                            i] = input_tensor_name

                        node_name_keep.insert(0, input_tensor_name)
                        _, _, _op = op_def_lib()._apply_op_helper(
                            "Placeholder",
                            dtype=input_node_type,
                            shape=input_shape,
                            name=input_tensor_name)
                        name_to_node[input_tensor_name] = _op.node_def
                        desc = dict()
                        desc['name'] = input_tensor_name
                        desc['shape'] = input_shape
                        desc['dtype'] = input_node_type
                        input_description.append(copy.deepcopy(desc))
        elif name_to_node_backup[per_node_name].op == 'Placeholder':
            # special case, the input node is a placeholder, \
            # add it directly to the input_description
            input_shape = tf.compat.v1.graph_util.tensor_shape_from_node_def_name(
                graph, name_to_node[per_node_name].name)

            if not input_shape:
                return queue.put([None, None])

            input_shape = input_shape_fix(input_shape)

            input_node_type = getattr(
                name_to_node[per_node_name].attr['dtype'], 'type')

            desc = dict()
            desc['name'] = name_to_node[per_node_name].name
            desc['shape'] = input_shape
            desc['dtype'] = input_node_type
            input_description.append(copy.deepcopy(desc))
        else:
            continue

    # generate the sub graphdef
    sub_graphdef = graph_pb2.GraphDef()
    for per_node_name in node_name_keep:
        sub_graphdef.node.extend([copy.deepcopy(name_to_node[per_node_name])])

    # return sub_graphdef,input_description
    queue.put([sub_graphdef, input_description])
