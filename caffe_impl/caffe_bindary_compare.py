"""
    This module is used  for caffe bindary search
"""
import os
import copy
import struct
import numpy as np
from google.protobuf import text_format
from .caffe_util import CaffeUtil


class LayerNode(object):
    """
        This class used to store graph node information
        parent: used to store node's parent information
        child:used to store node's child information
        name:used to store node's name
    """

    def __init__(self, name, layer_tpye):
        self.parent = []
        self.child = []
        self.name = name
        self.type = layer_tpye


class CaffeLayer(object):
    def __init__(self, name, op, bottoms, tops):
        self.name = name
        self.op = op
        self.bottom_blobs = list()
        self.top_blobs = list()
        self.add_bottoms(bottoms)
        self.add_tops(tops)

    def add_tops(self, tops):
        if not tops:
            return
        for top in tops:
            self.top_blobs.append(top)

    def add_bottoms(self, bottoms):
        if not bottoms:
            return
        for bottom in bottoms:
            self.bottom_blobs.append(bottom)


class CaffeBlob(object):
    def __init__(self, name, input_layer, output_layer):
        self.blob_name = name
        self.input_layer = input_layer
        self.output_layer = output_layer

    def add_input(self, in_layer):
        self.input_layer = in_layer

    def add_output(self, out_layer):
        self.output_layer = out_layer


class SubNetExtract(object):
    def __init__(self):
        self.model = None
        self.input_blob = set()
        self.valid_layer = set()
        self.graph = dict()
        self.blob_map = dict()

    def clear(self):
        self.model = None
        self.input_blob.clear()
        self.valid_layer.clear()
        self.graph.clear()
        self.blob_map.clear()

    def build_graph(self):
        for layer in self.model.layer:
            bottom_list = list()
            for name in layer.bottom:
                bottom_list.append(name)
                blob = self.blob_map.get(name)
                if not blob:
                    self.blob_map[name] = CaffeBlob(name, None, layer.name)
                else:
                    blob.add_output(layer.name)
                    self.blob_map[name] = blob

            top_list = list()
            for name in layer.top:
                top_list.append(name)
                blob = self.blob_map.get(name)
                if not blob:
                    self.blob_map[name] = CaffeBlob(name, layer.name, None)
                else:
                    blob.add_input(layer.name)
            self.graph[layer.name] = CaffeLayer(layer.name, layer.type,
                                                bottom_list, top_list)

    def get_input_blob(self, input_layers):
        for name in input_layers:
            graph_node = self.graph[name]
            if graph_node.op == 'Input':
                for blob_name in graph_node.top_blobs:
                    self.input_blob.add(blob_name)
            else:
                for blob_name in graph_node.bottom_blobs:
                    layer_name = self.blob_map[blob_name].input_layer
                    if layer_name in self.valid_layer:
                        continue
                    self.input_blob.add(blob_name)
        return True

    def get_valid_layer(self, node_set):
        for node in node_set:
            graph_node = self.graph[node]
            if graph_node.op == 'Input':
                continue
            self.valid_layer.add(node)

    def extract_subnet(
            self,
            input_layers,
            output_layers,
            model,
            weight,
            node_set=None):
        self.clear()
        import caffe
        from caffe.proto import caffe_pb2
        self.model = copy.deepcopy(model)
        self.build_graph()
        self.get_valid_layer(node_set)
        self.get_input_blob(input_layers)

        model_tmp_path = './tmp_model.prototxt'
        CaffeUtil.save_prototxt(self.model, model_tmp_path)
        complete_net = caffe.Net(model_tmp_path, weight, caffe.TEST)
        proto = caffe_pb2.NetParameter()
        input_desc = list()
        # create input data layers
        index = 0
        for blob_name in self.input_blob:
            new_layer = proto.layer.add()
            new_name = 'top_' + str(index)
            new_layer.name = new_name
            new_layer.top.append(blob_name)
            new_layer.type = 'Input'
            blob_shape = caffe_pb2.BlobShape()
            shape_desc = list()
            for dim in complete_net.blobs[blob_name].data.shape:
                blob_shape.dim.append(dim)
                shape_desc.append(dim)
            new_layer.input_param.shape.append(blob_shape)
            desc = dict()
            desc['name'] = blob_name
            desc['shape'] = shape_desc
            desc['dtype'] = 'float32'
            input_desc.append(desc)
            index += 1

        for layer in self.model.layer:
            if layer.name in self.valid_layer:
                proto.layer.append(layer)
        CaffeUtil.remove_file(model_tmp_path)
        return input_desc, proto


class Bisection(object):
    """
        This class used to bindary search caffe net
        input: used to store caffe net input value
        head:  used to store graph head node imformation
        node_list: used to store all caffe node information
        left_node: used to store left node which not belong net1 or net2
        longest_path: used to store the longest path
        net1_all-node:used to store all net1 node
        net2_all_node:used to store all net2 node
        net1_input_node:used to store net1 all input node
        net2_input_node:used to store net2 all input node
        net1_output_node: used to store net2 all output node
        net2_output_node: used to store net2 all output
        prototxt_net: used to store caffe prototxt net
        caffemodel_net: used to store caffe model net
    """

    def __init__(self, ):
        self.left_node = []
        self.net1_all_node = []
        self.net1_input_node = []
        self.net1_output_node = []
        self.net2_all_node = []
        self.net2_input_node = []
        self.net2_output_node = []

    def blob_split(self, model_path, weight_path):
        """
            This function used to change layer which top name not equal layer name
        """
        import caffe
        from caffe.proto import caffe_pb2
        caffe_layer_type_list = []

        prototxt_net = caffe_pb2.NetParameter()
        try:
            for layer_type in caffe.layer_type_list():
                caffe_layer_type_list.append(layer_type)

            with open(model_path, 'r') as file_read:
                text_format.Merge(file_read.read(), prototxt_net)

            CaffeUtil.update_graph(prototxt_net)
            with open("new.prototxt", 'w') as file_read:
                file_read.write(text_format.MessageToString(prototxt_net))

            caffemodel_net = caffe.Net("new.prototxt", weight_path,
                                       caffe.TEST)
            os.remove("new.prototxt")
            for i in range(len(prototxt_net.layer)):
                if prototxt_net.layer[i].type not in caffe_layer_type_list:
                    print("Given caffe model is not supported")
                    return 0
            return 1, prototxt_net, caffemodel_net
        except IOError as except_error:
            print("Given caffe model is not supported: {es}".format(
                es=str(except_error)))
            return 0, None, None

    def save_net_file(self, net_prototxt, weight_path, file_name,
                      sub_model):
        """
            This function used to save sub prototxt net and sub caffemodel_net
        """
        if not os.path.exists(sub_model):
            os.mkdir(sub_model)

        import caffe
        prototxt_path = os.path.join(sub_model, file_name + ".prototxt")
        with open(prototxt_path, 'w') as file_read:
            file_read.write(text_format.MessageToString(net_prototxt))

        caffemodel_net = caffe.Net(prototxt_path, weight_path, caffe.TEST)
        sub_model_path = os.path.join(sub_model, file_name + ".caffemodel")
        caffemodel_net.save(sub_model_path)

    def create_goldden_data(self, input_info, path_list, output_node,
                            filename):
        """
            This function used to create gloden data
        """
        import caffe

        input_name_list = []
        input_data_list = {}
        net = caffe.Net(path_list[0], path_list[1], caffe.TEST)

        current_input_path = os.path.join(path_list[2], filename)
        if not os.path.exists(current_input_path):
            os.mkdir(current_input_path)

        model_input_info = ""
        for inf in input_info:
            input_data = np.random.uniform(1, 100, inf['shape']).astype(
                np.float32)

            if not model_input_info:
                model_input_info += inf['name'] + ":" + ','.join(
                    (str(i) for i in inf['shape']))
            else:
                model_input_info += ";" + inf['name'] + ":" + ','.join(
                    (str(i) for i in inf['shape']))
            input_name_list.append(inf['name'])
            input_data_list[inf['name']] = input_data
            net.blobs[inf['name']].data[...] = input_data

        input_data_name = os.path.join(current_input_path,
                                       "".join(input_name_list) + ".bin")

        with open(input_data_name, 'wb') as file_write:
            file_write.write(struct.pack('f', len(input_name_list)))
            for name in input_name_list:
                file_write.write(
                    struct.pack('f', input_data_list[name].nbytes))

            for name in input_name_list:
                file_write.write(input_data_list[name])

        input_desc = os.path.join(current_input_path, "input_desc.txt")
        with open(input_desc, 'w') as file_write:
            file_write.write(input_data_name)

        net.forward()
        for node in output_node:
            output_data = net.blobs[node].data[...]

            with open(os.path.join(current_input_path, node + '.bin'),
                      'wb') as file_read:
                file_read.write(output_data.tobytes())

            with open(os.path.join(current_input_path, 'output_desc.txt'),
                      'a') as file_read:
                file_read.write(
                    os.path.join(current_input_path, node + '.bin\n'))

        return filename, model_input_info, input_data_name

    def create_net(self, net):
        """
            This function used to create caffe net
        """
        node_list = {}
        head = {}
        for i in range(len(net.layer)):
            flag = False
            node = LayerNode(str(net.layer[i].name), str(net.layer[i].type))
            name = ""
            if len(net.layer[i].bottom) != 0:
                for bottom in net.layer[i].bottom:
                    for j in range(i):
                        if bottom in net.layer[j].top:
                            name = net.layer[j].name
                            flag = True

                    if flag:
                        if node_list.get(name):
                            node_list[name].child.append(node.name)
                            node.parent.append(name)
                    else:
                        head[node.name] = node
            else:
                head[node.name] = node
            node_list[str(net.layer[i].name)] = node
        return node_list, head

    def create_path(self, node_list, head):
        """
            This function used to create caffe path
        """
        longest_length = 0
        longest_index = 0
        path_list = []
        for key, value in head.items():
            path = []
            path.append(key)
            path1 = list(path)
            child_path = self.dfs_create_path(value, path1, node_list)
            path_list.extend(child_path)
            path.remove(key)

        for path in path_list:

            if len(path) > longest_length:
                longest_length = len(path)
                longest_index = path_list.index(path)
        return len(path_list[longest_index]), path_list[longest_index]

    def dfs_create_path(self, node, path, node_list):
        """
            This function dfs create path
        """
        child_path = []
        if len(node.child) == 0:
            child_path.append(path)
        else:
            for child_node in node.child:
                path.append(child_node)
                path1 = list(path)
                return_path = self.dfs_create_path(node_list[child_node],
                                                   path1, node_list)
                child_path.extend(return_path)
                path.remove(child_node)
        return child_path

    def graph_bisection(self, len_longest_path, node_list, longest_path):
        """
            This function used to graph bindary
        """
        if len_longest_path != 2:
            half_len_longest_path = len_longest_path // 2

            bisection_node_name = longest_path[half_len_longest_path]
            for node in node_list[bisection_node_name].child:
                if node_list[node].type == "Eltwise":
                    if half_len_longest_path > 0:
                        half_len_longest_path = half_len_longest_path - 1
                        bisection_node_name = longest_path[
                            half_len_longest_path]
                    else:
                        return False
            bisection_node = node_list[bisection_node_name]

            self.net1_all_node.append(bisection_node_name)
            self.parent_search(bisection_node, node_list)
            self.child_search(bisection_node, node_list)

            for node_name, value in node_list.items():
                if node_name not in self.net1_all_node and \
                        node_name not in self.net2_all_node:
                    self.left_node.append(value)
            return_value = self.append_left(self.left_node, self.net1_all_node,
                                            self.net2_all_node)
            if return_value:
                self.append_input_output(node_list)
            else:
                return False
            return True
        else:
            if node_list[longest_path[0]].type != "Input":
                self.net1_input_node.append(longest_path[0])
                self.net1_output_node.append(longest_path[0])

                self.net2_input_node.append(longest_path[1])
                self.net2_output_node.append(longest_path[1])

                self.net1_all_node.append(longest_path[0])
                self.net2_all_node.append(longest_path[1])
                return True
            return False

    def append_input_output(self, node_list):
        """
            This function used to append input output node
        """
        for node in self.net1_all_node:
            child_list = []
            child_list.extend(node_list[node].child)
            for child_node in child_list:
                if child_node in self.net2_all_node:
                    node_list[node].child.remove(child_node)
                    node_list[child_node].parent.remove(node)
                    if child_node not in self.net2_input_node:
                        self.net2_input_node.append(child_node)
                    if len(node_list[node].child) == 0 and \
                            node not in self.net1_output_node:
                        self.net1_output_node.append(node)
            if len(node_list[node].child) == 0:
                if node not in self.net1_output_node:
                    self.net1_output_node.append(node)

            if len(node_list[node].parent) == 0:
                if node not in self.net1_input_node:
                    self.net1_input_node.append(node)

        for node in self.net2_all_node:
            if len(node_list[node].child) == 0:
                if node not in self.net2_output_node:
                    self.net2_output_node.append(node)

            if len(node_list[node].parent) == 0:
                if node not in self.net2_input_node:
                    self.net2_input_node.append(node)

    def append_left(self, left_node, net1_all_node, net2_all_node):
        """
            This function used to append left node
        """
        while True:
            flag = False
            for node in left_node:
                for child_node in node.child:
                    if child_node in net1_all_node:
                        net1_all_node.append(node.name)
                        flag = True
                        break

                    if child_node in net2_all_node:
                        net2_all_node.append(node.name)
                        flag = True
                        break
                if flag:
                    left_node.remove(node)
                    break
                for parent_node in node.parent:
                    if parent_node in net1_all_node:
                        net1_all_node.append(node.name)
                        flag = True
                        break

                    if parent_node in net2_all_node:
                        net2_all_node.append(node.name)
                        flag = True
                        break
                if flag:
                    left_node.remove(node)
                    break
            if not flag or len(self.left_node) == 0:
                if not flag and len(self.left_node) != 0:
                    return False
                return True

    def parent_search(self, node, node_list):
        """
            This function used to parent search
        """
        if len(node.parent) == 0:
            if node.name not in self.net1_input_node:
                self.net1_input_node.append(node.name)
            return

        if len(node.child) == 0:
            if node.name not in self.net1_output_node:
                self.net1_output_node.append(node.name)

        for parent_node in node.parent:
            self.net1_all_node.append(parent_node)
            self.parent_search(node_list[parent_node], node_list)

    def child_search(self, node, node_list):
        """
            This function used to child search
        """
        if len(node.parent) == 0:
            if node.name not in self.net2_input_node:
                self.net2_input_node.append(node.name)

        if len(node.child) == 0:
            if node.name not in self.net2_output_node:
                self.net2_output_node.append(node.name)
            return

        for child_node in node.child:
            self.net2_all_node.append(child_node)
            self.child_search(node_list[child_node], node_list)

    def clear(self, ):
        """
            This function used to clear enviroment
        """
        self.net1_output_node.clear()
        self.net1_all_node.clear()

        self.net2_output_node.clear()
        self.net2_all_node.clear()

        self.net2_input_node.clear()
        self.net1_input_node.clear()

        self.left_node.clear()
