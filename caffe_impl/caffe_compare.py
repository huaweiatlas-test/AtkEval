import os
import shutil
import sys
from utils.common_info import CommonInfo
from utils.convert2davinci import convert_model
from utils.davinci_run import davinci_run
from utils.precision_compare import precision_compare
from . import caffe_bindary_compare
from . import caffe_subnet_extract
from . import caffe_successive_compare
from . import caffe_util


class CaffeCompare(object):
    ''' class of precision compare of caffe model
    '''

    def __init__(self, user_config):
        self.ddk_path = os.path.abspath(user_config['ddk_path'])
        self.caffe_path = os.path.abspath(user_config['caffe_path'])
        self.model_path = os.path.abspath(user_config['model_path'])
        self.weight_path = os.path.abspath(user_config['weight_path'])
        self.result_path = os.path.abspath(user_config['result_path'])
        self.mode_number = user_config['mode_number']
        self.info_dict = {}
        if self.mode_number == '2':
            self.start_nodes = user_config['start_nodes']
            self.end_nodes = user_config['end_nodes']

        self.base_path = CommonInfo.get_tmp_base_path()
        self.sub_model_path = CommonInfo.get_sub_model_path()
        self.golden_data_path = CommonInfo.get_golden_data_path()
        self.davinci_model_path = CommonInfo.get_davinci_model_path()
        self.davinci_data_path = CommonInfo.get_davinci_data_path()
        self.black_list = CommonInfo.get_caffe_black_list()
        self.tmp_path_clear()
        self.insert_pycaffe_path()
        os.environ['GLOG_minloglevel'] = '3'

    def insert_pycaffe_path(self):
        ''' insert pycaffe path to system path
        '''
        sys.path.insert(1, self.caffe_path)

    def tmp_path_clear(self):
        ''' create .tmp folder to save models and data
        '''
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
        os.mkdir(self.base_path)
        os.mkdir(self.sub_model_path)
        os.mkdir(self.golden_data_path)
        os.mkdir(self.davinci_model_path)
        os.mkdir(self.davinci_data_path)

        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        self.info_dict.clear()

    def successive_compare(self):
        ''' single-layer precision compare
        '''
        from caffe.proto import caffe_pb2
        proto = caffe_pb2.NetParameter()
        compare = caffe_successive_compare.SuccesiveCompare(proto,
                                                            self.model_path,
                                                            self.weight_path)
        self.info_dict = compare.successive_compare(self.golden_data_path,
                                                    self.sub_model_path)
        self.compare_process()

    def specific_compare(self):
        ''' sub-model precision compare
        '''
        import caffe
        from google.protobuf import text_format
        compare = caffe_subnet_extract.CaffeSubnetExtract()
        net = compare.read_prototxt(self.model_path)

        # begin
        model_name = "sub_model"
        caffe_util.CaffeUtil.update_graph(net)
        with open("comp.prototxt", 'w') as model_file:
            model_file.write(text_format.MessageToString(net))
        net1 = caffe.Net("comp.prototxt", self.weight_path, caffe.TEST)
        os.remove("comp.prototxt")
        try:
            data_input, final_net, final_net2 = compare.caffe_subnet_extract(
                self.start_nodes, self.end_nodes, net, net1)
        except RuntimeError as error_info:
            print(str(error_info))
            exit()

        prototxt_path = os.path.join(self.sub_model_path,
                                     model_name + ".prototxt")
        caffemodel_path = os.path.join(self.sub_model_path,
                                       model_name + ".caffemodel")
        with open(prototxt_path, 'w') as model_file:
            model_file.write(text_format.MessageToString(final_net))
        final_net2.save(caffemodel_path)
        compare.create_goldden_data(final_net2, self.golden_data_path,
                                    model_name, data_input, self.end_nodes)
        self.compare_process()

    def binary_compare(self):
        ''' model bindary precision compare
        '''
        bisect = caffe_bindary_compare.Bisection()
        caffe_layer = caffe_bindary_compare.SubNetExtract()
        read_flag, net1, net2 = bisect.blob_split(
            self.model_path,
            self.weight_path)
        node_list, head = bisect.create_net(net1)
        time = 1
        if read_flag:
            while True:
                len_longest_path, longest_path = bisect.create_path(node_list,
                                                                    head)
                return_flag = bisect.graph_bisection(len_longest_path,
                                                     node_list, longest_path)
                if return_flag:
                    self.tmp_path_clear()
                    data_input1_shape, prototxt_net1 = \
                        caffe_layer.extract_subnet(
                            bisect.net1_input_node, bisect.net1_output_node,
                            net1,
                            self.weight_path,
                            bisect.net1_all_node)
                    data_input2_shape, prototxt_net2 = \
                        caffe_layer.extract_subnet(
                            bisect.net2_input_node, bisect.net2_output_node,
                            net1,
                            self.weight_path,
                            bisect.net2_all_node)
                    bisect.save_net_file(prototxt_net1, self.weight_path,
                                         "".join(bisect.net1_input_node),
                                         self.sub_model_path)
                    bisect.save_net_file(prototxt_net2, self.weight_path,
                                         "".join(bisect.net2_input_node),
                                         self.sub_model_path)

                    prototxt_path1 = self.sub_model_path + '/' + "".join(
                        bisect.net1_input_node) + ".prototxt"
                    caffemodel_path1 = self.sub_model_path + '/' + "".join(
                        bisect.net1_input_node) + ".caffemodel"
                    filename, model_input_info, _ = \
                        bisect.create_goldden_data(
                            data_input1_shape,
                            [prototxt_path1, caffemodel_path1,
                             self.golden_data_path],
                            bisect.net1_output_node,
                            "".join(bisect.net1_input_node))
                    self.info_dict[filename] = model_input_info

                    prototxt_path2 = self.sub_model_path + '/' + "".join(
                        bisect.net2_input_node) + ".prototxt"
                    caffemodel_path2 = self.sub_model_path + '/' + "".join(
                        bisect.net2_input_node) + ".caffemodel"
                    filename, model_input_info, _ = \
                        bisect.create_goldden_data(
                            data_input2_shape,
                            [prototxt_path2,
                             caffemodel_path2,
                             self.golden_data_path],
                            bisect.net2_output_node,
                            "".join(
                                bisect.net2_input_node))
                    self.info_dict[filename] = model_input_info

                    compare_result = self.compare_process()

                    net1_compare_result = 0
                    net2_compare_result = 0
                    net1_count = 0
                    net2_count = 0
                    for item, value in compare_result.items():
                        if item in bisect.net1_output_node:
                            net1_compare_result += value[0]
                            net1_count = net1_count + 1
                        else:
                            net2_compare_result += value[0]
                            net2_count = net2_count + 1

                    if net1_count != 0:
                        net1_compare_result = net1_compare_result / net1_count

                    if net2_count != 0:
                        net2_compare_result = net2_compare_result / net2_count

                    if len(bisect.net1_all_node) == 1 or len(
                            bisect.net2_all_node) == 1:
                        break

                    if net1_compare_result > net2_compare_result:
                        bisect.clear()
                        node_list, head = bisect.create_net(prototxt_net1)
                        net1 = prototxt_net1
                    else:
                        bisect.clear()
                        node_list, head = bisect.create_net(prototxt_net2)
                        net1 = prototxt_net2
                    time = time + 1
                else:
                    break
            print(
                "-------------------------------The final sub net nodes---------------------------------")
            for key, node in node_list.items():
                if node.type != "Input":
                    print("node name: {}".format(node.name))

    def compare_process(self):
        ''' post-process: convert to D-model, D-inference, result compare
        '''
        print('#################Begin to run OMG...#############')
        model_name_lsit = convert_model(self.ddk_path, self.sub_model_path,
                                        'caffe', self.davinci_model_path,
                                        input_dict=self.info_dict)

        print('#################Begin to run Davinci models...#############')
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
        ''' run precision compare based on different mode
        '''
        if self.mode_number == '1':
            self.successive_compare()
        elif self.mode_number == '2':
            self.specific_compare()
        elif self.mode_number == '3':
            self.binary_compare()
        else:
            print('unsupported mode, exit')
            exit()
