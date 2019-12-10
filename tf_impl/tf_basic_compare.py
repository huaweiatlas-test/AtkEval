import numpy as np
import os
import shutil
import tensorflow as tf
from multiprocessing import Process, Queue
from tensorflow.core.framework import types_pb2
from utils.common_info import CommonInfo
from utils.convert2davinci import convert_model
from utils.davinci_run import davinci_run
from utils.precision_compare import precision_compare

from . import data_save
from . import tf_binary_split
from . import tf_subnet_extract


class TensorflowBasicCompare:
    '''basic class for tensorflow compare,this is the only
    tensorflow interface used by the main function'''
    def __init__(self, user_config_dict):

        self.ddk_path = user_config_dict['ddk_path']
        self.model_path = user_config_dict['model_path']
        self.result_path = user_config_dict['result_path']
        self.mode_number = user_config_dict['mode_number']
        if int(self.mode_number) == 2:
            self.start_nodes = user_config_dict['start_nodes']
            self.end_nodes = user_config_dict['end_nodes']

        self.base_path = CommonInfo.get_tmp_base_path()
        self.sub_model_path = CommonInfo.get_sub_model_path()
        self.golden_data_path = CommonInfo.get_golden_data_path()
        self.davinci_model_path = CommonInfo.get_davinci_model_path()
        self.davinci_data_path = CommonInfo.get_davinci_data_path()
        self.black_list = CommonInfo.get_tf_black_list()

        self.tmp_path_clear()

    def tmp_path_clear(self):
        '''clear the temporary files'''
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
            os.mkdir(self.base_path)
        else:
            os.mkdir(self.base_path)
        os.mkdir(self.sub_model_path)
        os.mkdir(self.golden_data_path)
        os.mkdir(self.davinci_model_path)
        os.mkdir(self.davinci_data_path)

        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)

    def tensorshape2shape(self, tensor_shape):
        ''' convert the tf.TensorShape to list
        Args:
            tensor_shape: tf.TensorShape type, represents the shape of a tensor
        Return:
            shape_convert: a list whose element is dim value
        '''

        shape_old = tensor_shape.as_list()
        shape_convert = []
        for idx, dim_shape in enumerate(shape_old):
            if dim_shape is None:
                if idx == 0:
                    shape_convert.append(1)
                else:
                    shape_convert.append(64)
            else:
                shape_convert.append(dim_shape)
        return shape_convert

    def random_input_data_gen(self, input_description):
        ''' genetate random input data given the input type and shape
        Args:
            input_description: a list whose element is a dict, and the dict \
            contain the shape and type of the input
        Return:
            feed_dict: a dict. key is the input tensor name, ending with ':0'.\
            value is the numpy array
        '''
        feed_dict = dict()
        for input_desc in input_description:
            data_shape = self.tensorshape2shape(input_desc['shape'])
            data_shape_str = [str(s) for s in data_shape]
            if input_desc['dtype'] == types_pb2.DT_FLOAT:
                data_tmp = eval('np.random.rand(' + ','.join(data_shape_str) +
                                ')')
                data_tmp = data_tmp.astype(np.float32)
            elif input_desc['dtype'] == types_pb2.DT_INT32:
                data_tmp = np.random.randint(64, size=tuple(data_shape))
                data_tmp = data_tmp.astype(np.int32)
            elif input_desc['dtype'] == types_pb2.DT_INT64:
                data_tmp = np.random.randint(64, size=tuple(data_shape))
                data_tmp = data_tmp.astype(np.int64)
            elif input_desc['dtype'] == types_pb2.DT_UINT8:
                data_tmp = np.random.randint(64, size=tuple(data_shape))
                data_tmp = data_tmp.astype(np.uint8)
            else:
                print('Unsupported input type, skip!')
                return -1
            feed_dict[input_desc['name'] + ':0'] = data_tmp
        return feed_dict

    class TensorflowPredictor:
        '''
        tensorflow inference class.
        Load the graph_def and do inference
        '''
        def __init__(self, graph_def):
            self.graph_def = graph_def
            self.model_load()

        def model_load(self):
            ''' Load model from the graph_def object,\
             it can be a single op model'''
            if self.graph_def is None:
                raise RuntimeError(
                    'Cannot find inference graph in tar archive.')
            self.graph = tf.Graph()
            with self.graph.as_default():
                try:
                    tf.import_graph_def(self.graph_def, name='')
                except:
                    raise RuntimeError('unsupported model')
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(graph=self.graph, config=config)

        def model_infer(self, input_dict, output_node):
            ''' do inference
            Args:
                input_dict: a dict including all the inputs of the model, key \
                is the node name and value is numpy array data
                output_node: a list contains the output tensor names of the \
                output nodes, one node one output tensor
            Return:
                res_dict: a dict, key is the node name and value is the numpy \
                array output data
            '''
            output_tensor_name = []
            for node in output_node:
                output_tensor_name.append(node + ':0')
            res_lsit = self.sess.run(output_tensor_name, feed_dict=input_dict)
            res_dict = dict()
            for idx, node in enumerate(output_node):
                res_dict[node] = res_lsit[idx]
            return res_dict

    def successive_compare(self):

        with tf.gfile.FastGFile(self.model_path, 'rb') as pb_f:
            global_graph_def = tf.compat.v1.GraphDef.FromString(pb_f.read())
        global_graph = tf.Graph()
        with global_graph.as_default():
            tf.import_graph_def(global_graph_def, name='')

        feed_dict_collection = []
        res_dict_collection = []

        for node in global_graph_def.node:

            if node.op in self.black_list:
                continue

            io_queue = Queue()
            pid = Process(target=tf_subnet_extract.tf_subnet_extract,
                          args=(global_graph_def, global_graph, [node.name],
                                [node.name], io_queue, None))

            pid.start()
            sub_graphdef, input_description = io_queue.get()
            pid.join()

            if not sub_graphdef:
                continue

            feed_dict = self.random_input_data_gen(input_description)
            if not feed_dict:
                continue

            tfp = self.TensorflowPredictor(sub_graphdef)
            res_dict = tfp.model_infer(feed_dict, [node.name])
            pb_name = node.name.replace('/', '_')

            with tf.io.gfile.GFile(
                    os.path.join(self.sub_model_path, pb_name + '.pb'),
                    "wb") as pb_f:
                pb_f.write(sub_graphdef.SerializeToString())
            feed_dict['model_name'] = pb_name
            res_dict['model_name'] = pb_name
            feed_dict_collection.append(feed_dict)
            res_dict_collection.append(res_dict)
            print('node:', node.name, ' runs on tensorflow SUCCESS')

        self.compare_process(feed_dict=feed_dict_collection,
                             res_dict=res_dict_collection)

    def specific_compare(self):
        '''compare the specific sub model'''
        with tf.gfile.FastGFile(self.model_path, 'rb') as pb_f:
            global_graph_def = tf.compat.v1.GraphDef.FromString(pb_f.read())
        global_graph = tf.Graph()
        with global_graph.as_default():
            tf.import_graph_def(global_graph_def, name='')

        node_map = dict()
        for node in global_graph_def.node:
            node_map[node.name] = node.op

        for node_list in self.start_nodes, self.end_nodes:
            for node_name in node_list:
                node_op = node_map.get(node_name)
                if not node_op:
                    print(
                        '{} is not in graph. Please check.'.format(node_name))
                    return
                if CommonInfo.is_skip_tf_op(node_op):
                    print('{} is in black_list, exit'.format(node_name))
                    return

        io_queue = Queue()
        pid = Process(target=tf_subnet_extract.tf_subnet_extract,
                      args=(global_graph_def, global_graph, self.start_nodes,
                            self.end_nodes, io_queue, None))
        pid.start()
        sub_graphdef, input_description = io_queue.get()
        pid.join()
        if not sub_graphdef:
            return

        feed_dict = self.random_input_data_gen(input_description)
        if not feed_dict:
            return
        tfp = self.TensorflowPredictor(sub_graphdef)
        res_dict = tfp.model_infer(feed_dict, self.end_nodes)

        pb_name = 'sub_graph'
        with tf.io.gfile.GFile(
                os.path.join(self.sub_model_path, pb_name + '.pb'),
                "wb") as pb_f:
            pb_f.write(sub_graphdef.SerializeToString())
        feed_dict['model_name'] = pb_name
        res_dict['model_name'] = pb_name
        feed_dict_collection = []
        res_dict_collection = []
        feed_dict_collection.append(feed_dict)
        res_dict_collection.append(res_dict)
        print('sub graph runs on tensorflow SUCCESS')
        self.compare_process(feed_dict=feed_dict_collection,
                             res_dict=res_dict_collection)

    def binary_compare(self):
        '''use dichotomy to find the op with worst precision'''
        model_path = self.model_path
        index = 0
        distance_dict = {}
        while model_path:
            with tf.gfile.FastGFile(model_path, 'rb') as pb_f:
                proto = tf.compat.v1.GraphDef.FromString(pb_f.read())
            global_graph = tf.Graph()
            with global_graph.as_default():
                tf.import_graph_def(proto, name='')
            top_graph, bottom_graph = tf_binary_split.split_graph(proto)
            if not top_graph or not bottom_graph:
                break
            self.tmp_path_clear()
            queue = Queue()
            feed_dict_collection = []
            res_dict_collection = []
            for sub_graph in top_graph, bottom_graph:
                pid = Process(target=tf_subnet_extract.tf_subnet_extract,
                              args=(proto, global_graph,
                                    sub_graph.get_inputs(),
                                    sub_graph.get_outputs(), queue,
                                    sub_graph.get_nodes()))
                pid.start()
                sub_graph_def, input_description = queue.get()
                pid.join()

                feed_dict = self.random_input_data_gen(input_description)
                if not feed_dict:
                    print('random input data gen error.')
                    return
                tfp = self.TensorflowPredictor(sub_graph_def)
                res_dict = tfp.model_infer(feed_dict, sub_graph.get_outputs())
                if not res_dict:
                    print('tensorflow inference error.')
                    return
                pb_name = sub_graph.model_name + '_' + str(index)
                sub_pb = os.path.join(self.sub_model_path, pb_name + '.pb')
                with tf.io.gfile.GFile(sub_pb, "wb") as pb_f:
                    pb_f.write(sub_graph_def.SerializeToString())
                feed_dict['model_name'] = pb_name
                res_dict['model_name'] = pb_name
                feed_dict_collection.append(feed_dict)
                res_dict_collection.append(res_dict)
            distance_dict = self.compare_process(
                feed_dict=feed_dict_collection, res_dict=res_dict_collection)

            next_model, _ = tf_binary_split.get_worse_model(
                distance_dict, top_graph, bottom_graph)
            if not next_model:
                print('[Error] There is no precision result. Please check.')
                return
            next_pb = next_model.model_name + '_' + str(index) + '.pb'
            model_path = os.path.join(self.sub_model_path, next_pb)
            if next_model.get_valid_node_num() <= 1:
                break
            index += 1
        print('You could check the sub-model {}'.format(model_path))
        submodel_nodes = tf_binary_split.get_model_nodes(model_path)
        print("#################The final sub net nodes#############")
        for name in submodel_nodes:
            print("node name: {}".format(name))

    def compare_process(self, feed_dict, res_dict):
        '''the shared process of different mode'''
        print('#################Begin to save data...#############')
        input_desc_collection = data_save.data_save(feed_dict, res_dict,
                                                    self.golden_data_path)

        print('#################Begin to run OMG...#############')
        model_name_lsit = convert_model(self.ddk_path, self.sub_model_path,
                                        'tensorflow', self.davinci_model_path,
                                        input_desc_collection)

        print('#################Begin to run DaVinci models...#############')
        valid_model_name_list = davinci_run(self.davinci_model_path,
                                            self.golden_data_path,
                                            self.davinci_data_path,
                                            model_name_lsit)

        print('#################Begin to compare...#############')
        precision_result = precision_compare(self.davinci_data_path,
                                             self.golden_data_path,
                                             valid_model_name_list,
                                             self.result_path)
        return precision_result

    def run(self):
        '''the entrance of the function'''
        if self.mode_number == '1':
            self.successive_compare()
        elif self.mode_number == '2':
            self.specific_compare()
        elif self.mode_number == '3':
            self.binary_compare()
        else:
            print('unsupported mode, exit')
            exit()
