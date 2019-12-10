# coding=UTF-8
import tensorflow as tf
from google.protobuf import text_format
from utils.common_info import CommonInfo


class SimpleModel:
    ''' model info class
    '''

    def __init__(self):
        self.model_name = ''
        self.nodes = set()
        self.inputs = set()
        self.outputs = set()
        self.valid_node_nums = 0

    def add_single_node(self, node):
        ''' add single graph node to model
        '''
        if node is None:
            return
        self.nodes.add(node)

    def add_node_set(self, node_set):
        ''' add graph nodes to model
        '''
        if node_set is None:
            return
        self.nodes = self.nodes.union(node_set)

    def set_model_name(self, name):
        ''' set model name
        '''
        self.model_name = name
        return

    def get_inputs(self):
        ''' get input nodes of model
        '''
        return list(self.inputs)

    def get_outputs(self):
        ''' get output nodes of model
        '''
        return list(self.outputs)

    def get_nodes(self):
        ''' get all nodes of model
        '''
        return list(self.nodes)

    def get_valid_node_num(self):
        ''' get the number of valid nodes
        '''
        return self.valid_node_nums

    def count_valid_node_num(self, valid_nodes_dict):
        ''' count the number of valid nodes
        '''
        count = 0
        for node_name in self.nodes:
            node = valid_nodes_dict.get(node_name)
            if not node:
                continue
            if CommonInfo.is_skip_tf_op(node.type):
                continue
            count += 1
        self.valid_node_nums = count

    def gen_io_nodes(self, valid_nodes_dict):
        ''' generate the input/output nodes
        '''
        self.inputs.clear()
        self.outputs.clear()
        for node_name in self.nodes:
            node = valid_nodes_dict.get(node_name)
            if not node:
                continue
            if not node.parent:
                self.inputs.add(node_name)
            for parent in node.parent:
                if parent not in self.nodes:
                    self.inputs.add(node_name)
                    continue
            if not node.child:
                self.outputs.add(node_name)
            for child in node.child:
                if child not in self.nodes:
                    self.outputs.add(node_name)
                    continue


class Node:
    ''' node class
    '''

    def __init__(self, name, op, input_nodes):
        self.name = name
        self.type = op
        self.parent = set()
        self.child = set()
        self.add_parents(input_nodes)

    def add_parents(self, nodes):
        ''' add parent nodes
        '''
        if nodes is None:
            return
        for node in nodes:
            self.parent.add(node)

    def add_child(self, node):
        ''' add child nodes
        '''
        if node is None:
            return
        self.child.add(node)


class Parser:
    ''' class of parse tensorflow graph
    '''

    def __init__(self):
        self.invalid = 'invalid'
        self.model_inputs = list()
        self.cross_node_map = set()
        self.valid_node_map = dict()
        self.complete_nodes = dict()
        self.models_name = ['top_model', 'bottom_model']

    @staticmethod
    def get_graph_def(pb):
        ''' get graph_def from pb file
        '''
        with open(pb, 'rb') as pb_file:
            proto = tf.compat.v1.GraphDef()
            proto.ParseFromString(pb_file.read())
        return proto

    @staticmethod
    def save_pbtxt(path, proto):
        ''' convert graph_def to txt
        '''
        with open(path, 'w') as pbtxt_file:
            pbtxt_file.write(text_format.MessageToString(proto))
            print('pbtxt path: ' + path)

    @staticmethod
    def save_pb(path, proto):
        ''' save graph_def as pb file
        '''
        with open(path, 'wb') as pb_file:
            pb_file.write(proto.SerializeToString())

    def get_cross_node_map(self):
        ''' generate cross nodes map
        '''
        for node in self.valid_node_map.values():
            if len(node.parent) > 1 or len(node.child) > 1:
                self.cross_node_map.add(node.name)

    def graph_builder(self, graph_def):
        ''' get model info from graph_def
        '''
        self.complete_nodes.clear()
        for node in graph_def.node:
            type = node.op
            if (node.op == 'Identity' and '_class' in node.attr.keys()) \
                    or node.op == 'Const':
                type = 'Weight'
            self.complete_nodes[node.name] = Node(node.name, type, node.input)

        del self.model_inputs[:]
        self.valid_node_map.clear()
        for node_name in self.complete_nodes:
            node = self.complete_nodes[node_name]
            if node.type == 'Weight':
                continue
            valid_inputs = []
            for pre_node in node.parent:
                parent = self.complete_nodes.get(pre_node, self.invalid)
                if parent != self.invalid and parent.type != 'Weight':
                    valid_inputs.append(pre_node)
            self.valid_node_map[node.name] = Node(node.name, node.type,
                                                  valid_inputs)
            if not node.parent:
                self.model_inputs.append(node.name)

    def update_graph_edge(self):
        ''' add edge of graph nodes
        '''
        for node_name in self.valid_node_map:
            node = self.valid_node_map[node_name]
            for pre_node in node.parent:
                parent = self.valid_node_map.get(pre_node, self.invalid)
                if parent != self.invalid and parent.type != 'Weight':
                    parent.add_child(node.name)
                    self.valid_node_map[parent.name] = parent
                else:
                    print('[Error] the input of node is invalid. \
                        Please check graph-builder')
                    print('current node {} --> input {}'.format(
                        node_name, parent.name))

    def parse_model(self, graph_def):
        ''' parse tensorflow model
        '''
        self.graph_builder(graph_def)
        self.update_graph_edge()
        self.get_cross_node_map()

    def get_longest_path(self, node_map, root_nodes):
        ''' find the longest path in model
        '''
        # {cur_node, parent_node} in longest path
        longest_node_pair = dict()
        cache_nodes_name = set()
        for input_name in root_nodes:
            longest_node_pair[input_name] = self.invalid
            cache_nodes_name.add(input_name)

        while cache_nodes_name:
            tmp_cache = set()
            for name in cache_nodes_name:
                node = node_map.get(name, self.invalid)
                if node == self.invalid:
                    continue
                for child in node.child:
                    longest_node_pair[child] = node.name
                    tmp_cache.add(child)
            if not tmp_cache:
                break
            cache_nodes_name = tmp_cache

        longest_path = []
        # If there are more than 1 longest paths, choose a path randomly.
        cur_node = cache_nodes_name.pop()
        while cur_node:
            pair_node = longest_node_pair.get(cur_node, self.invalid)
            if pair_node == self.invalid:
                break
            longest_path.append(cur_node)
            cur_node = pair_node
        longest_path.reverse()
        return longest_path

    def get_middle_node(self):
        ''' find the middle nodes of longest path
        '''
        longest_path = self.get_longest_path(self.valid_node_map,
                                             self.model_inputs)
        if not longest_path:
            return None, None
        cross_node_path = list()
        for name in longest_path:
            if name in self.cross_node_map:
                cross_node_path.append(name)

        top_end = None
        mid_index = 0
        # search split_node from corss_node_list firstly
        for node_path in cross_node_path, longest_path:
            mid_index = int(len(node_path) / 2) - 1
            while mid_index > 0:
                node_name = node_path[mid_index]
                top_end = self.valid_node_map.get(node_name)
                if CommonInfo.is_skip_tf_op(top_end.type):
                    mid_index -= 1
                else:
                    break
        if mid_index == 0:
            top_end = self.valid_node_map.get(longest_path[0])
            bottom_start = self.valid_node_map.get(longest_path[1])
        else:
            bottom_start = top_end
        return top_end, bottom_start

    def split_graph(self, top_end, bottom_start):
        ''' split model into two parts
        '''
        # search from high-level to low-level, to generate top_graph
        top_model = self.gen_simple_model(top_end, self.models_name[0], True)
        # search from low-level to high-level, to generate bottom_graph
        bottom_model = self.gen_simple_model(bottom_start, self.models_name[1],
                                             False)

        remaining_node = set()
        for name in self.complete_nodes:
            remaining_node.add(name)
        remaining_node.difference_update(top_model.nodes)
        remaining_node.difference_update(bottom_model.nodes)
        self.extend_simple_model(remaining_node, top_model)
        self.extend_simple_model(remaining_node, bottom_model)
        if remaining_node:
            print('[ERROR] there are some unattached nodes. \
                Please check graph builder.')
            print(remaining_node)
        return top_model, bottom_model

    def extend_simple_model(self, remaining_node, model):
        ''' extend single path model
        '''
        self.add_supple_node(remaining_node, model)
        model.count_valid_node_num(self.valid_node_map)
        self.add_weight_node(remaining_node, model)
        model.gen_io_nodes(self.valid_node_map)

    def gen_simple_model(self, mid_node, model_name, is_top_model):
        ''' generate single path model
        '''
        model = SimpleModel()
        model.set_model_name(model_name)
        cache_nodes = set()
        cache_nodes.add(mid_node.name)
        while cache_nodes:
            tmp_cache = set()
            for name in cache_nodes:
                model.add_single_node(name)
                node = self.valid_node_map.get(name, self.invalid)
                if is_top_model:
                    node_list = node.parent
                else:
                    node_list = node.child
                for tmp_node in node_list:
                    if tmp_node not in model.nodes:
                        tmp_cache.add(tmp_node)
            cache_nodes = tmp_cache
        return model

    def add_supple_node(self, remaining_node, model):
        ''' add supplement nodes to single path model
        '''
        while remaining_node:
            supple_node = set()
            for name in remaining_node:
                is_added = False
                node = self.valid_node_map.get(name, self.invalid)
                if node == self.invalid:
                    continue
                for parent in node.parent:
                    if parent in model.nodes or parent in supple_node:
                        supple_node.add(name)
                        is_added = True
                        break
                if is_added:
                    break
                for child in node.child:
                    if child in model.nodes or child in supple_node:
                        supple_node.add(name)
                        break
            if not supple_node:
                break
            remaining_node.difference_update(supple_node)
            model.add_node_set(supple_node)

    def add_weight_node(self, remaining_node, model):
        ''' add weight nodes to submodel
        '''
        next_node_set = model.nodes
        while remaining_node:
            cur_node_set = next_node_set.copy()
            next_node_set = set()
            for name in cur_node_set:
                node = self.complete_nodes.get(name, self.invalid)
                for parent in node.parent:
                    if parent in remaining_node:
                        next_node_set.add(parent)
            if not next_node_set:
                break
            remaining_node.difference_update(next_node_set)
            model.add_node_set(next_node_set)


def split_graph(proto):
    ''' split original graph into two parts
    '''
    parser = Parser()
    parser.parse_model(proto)
    top_end, bottom_start = parser.get_middle_node()
    if not top_end or not bottom_start:
        return None, None
    print('split node: ' + top_end.name)
    top_graph, bottom_graph = parser.split_graph(top_end, bottom_start)
    print('split finish...')
    return top_graph, bottom_graph


def get_worse_model(distance_dict, top_model, bottom_model):
    ''' calculate worse model
    '''
    worse_op_name = ''
    max_distance = 0.0
    for name in distance_dict:
        distance = distance_dict[name][0]
        if distance > max_distance:
            max_distance = distance
            worse_op_name = name

    for model in top_model, bottom_model:
        for output in model.outputs:
            name = output.replace('/', '_')
            if worse_op_name == name:
                return model, worse_op_name
    print('please check output nodes')
    return None, None


def get_model_nodes(model_path):
    ''' print the nodes of model
    '''
    with tf.gfile.FastGFile(model_path, 'rb') as pb_f:
        graph_def = tf.compat.v1.GraphDef.FromString(pb_f.read())
    node_list = list()
    for node in graph_def.node:
        type = node.op
        if (type == 'Identity' and '_class' in node.attr.keys()) \
                or type == 'Const' or type == 'Placeholder':
            continue
        node_list.append(node.name)
    return node_list
