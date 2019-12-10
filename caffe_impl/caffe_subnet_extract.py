# -*- coding: utf-8 -*
import copy
import os
import struct
import numpy as np
from google.protobuf import text_format
from utils import common_info


class CaffeSubnetExtract:
    ''' caffe sub-model compare class
    '''

    def __init__(self):
        self.golden_data_path = common_info.CommonInfo.get_golden_data_path()

    def check_layername(self, net, names):
        ''' check validity of layers
        Args:
            net: caffe net
            names: name of layers
        '''
        layer_name = [i.name for i in net.layer]
        if not set(names).issubset(layer_name):
            raise RuntimeError("Can't find the layer's name. Please Check "
                               "the prototxt and input / output name given")

    def check_network(self, prototxt_file):
        ''' check validity of network
        Args:
            prototxt_file: the path of model file
        '''
        # 避免子网输出节点无输入的情况
        net = self.read_prototxt(prototxt_file)
        layer_name = [i.name for i in net.layer]
        for i in range(len(net.layer)):
            if net.layer[i].type != "Input":
                bottom_names = net.layer[i].bottom
                if not set(bottom_names).issubset(layer_name):
                    raise RuntimeError(
                        "Network cannot be parsed! "
                        "Please cheak input / output nodes given.")
        return True

    def read_prototxt(self, prototxt_file):
        ''' get caffenet from prototxt file
        Args:
            prototxt_file: the path of model file
        Return:
            caffenet
        '''
        from caffe.proto import caffe_pb2
        net = caffe_pb2.NetParameter()
        with open(prototxt_file, 'rb') as model_file:
            text_format.Parse(model_file.read(), net)
        return net

    def save_prototxt(self, net, prototxt_file, data_input):
        ''' save submodel as prototxt
        Args:
            net: sub model
            prototxt_file: the path of model file
            data_input: input layer
        '''
        try:
            with open(prototxt_file, 'w') as model_file:
                model_file.write(data_input)
                # 避免出现prototxt中data input节点不以layer形式保存，
                # 而在新生成网络结构中无法删除的情况
                for ret in net.layer:
                    model_file.write('layer{ \n')
                    model_file.write(str(ret))
                    model_file.write('}\n')
        except IOError as except_error:
            print("Cannot save prototxt: {es}".format(es=str(except_error)))

    def save_modified_caffemodel(self, prototxt_file, caffemodel_file,
                                 new_caffemodel_file):
        ''' save submodel weight as caffemodel
        Args:
            net: sub model
            prototxt_file: the path of model file
            caffemodel_file: original weight file
            new_caffemodel_file: submodel weight file
        '''
        import caffe
        net = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)
        net.save(new_caffemodel_file)

    def get_layers_index(self, net, input_names):
        ''' get index of caffe-layers
        Args:
            net: caffe deploy net
            input_names: the name of certain layers
        '''
        index_lst = []
        for input_name in input_names:
            for i in range(len(net.layer)):
                if net.layer[i].name == input_name:
                    index_lst.append(i)
        return index_lst

    def get_index_bottom_and_top_names(self, net, custom_layer_name):
        ''' get model info of caffe-layer
        Args:
            net: caffe deploy net
            custom_layer_name: the name of custom_layer
        Return:
            index: index in caffe net
            bottom_names: the bottom blob of custom_layer
            top_names: the top blob of custom_layer
        '''
        bottom_names = None
        top_names = None
        index = -1
        for i in range(len(net.layer)):
            if net.layer[i].name == custom_layer_name:
                bottom_names = net.layer[i].bottom
                top_names = net.layer[i].top
                index = i
                break
        return index, bottom_names, top_names

    def del_layer(self, net, save_lay_index):
        ''' delete layers in caffe net
        Args:
            net: caffe deploy net
            save_lay_index: the index of saved layers
        '''
        save_lay_index.sort(reverse=True)
        for i in range(len(net.layer) - 1, -1, -1):
            if i not in save_lay_index:
                del net.layer[i]

    def find_same_blob(self, net, index, label):
        ''' get blob index of same blob in caffe net
        '''
        left, right = index, index
        set_blob_index = set()
        len_layer = len(net.layer)
        while right <= len_layer - 1 and net.layer[right].top == label:
            set_blob_index.add(right)
            right += 1
        while left >= 0 and net.layer[left].top == label:
            set_blob_index.add(left)
            left -= 1
        return set_blob_index

    def find_layer_forward(self, net, output_name):
        ''' store layer index of submodel in caffe net
        '''
        save_layer_dict = {}
        for node in output_name:
            index, bottom_names, top_names = \
                self.get_index_bottom_and_top_names(net, node)
            save_layer_dict[node] = index
            if index and bottom_names and top_names:
                # 可能有些层没有bottom name，也有可能有的层bottom是data
                # 但input层命名非data 因此增加判断
                for bottom_name in bottom_names:
                    index1, _, _ = self.get_index_bottom_and_top_names(
                        net, bottom_name)
                    save_layer_dict[bottom_name] = index1
        while True:
            find_node_name = {}
            i = 0
            while i < len(net.layer):
                if net.layer[i].name in save_layer_dict:
                    for node in save_layer_dict:
                        index, bottom_names, top_names = \
                            self.get_index_bottom_and_top_names(net, node)
                        if index and bottom_names and top_names:
                            for bottom_name in bottom_names:
                                if bottom_name not in save_layer_dict:
                                    index1, _, _ = \
                                        self.get_index_bottom_and_top_names(
                                            net, bottom_name)
                                    find_node_name[bottom_name] = index1
                i += 1
            if find_node_name:
                save_layer_dict.update(find_node_name)
            else:
                break

        save_lay_index = [i for i in save_layer_dict.values()]
        save_lay_index.sort(reverse=True)
        return save_lay_index

    def get_datainput(self, table):
        ''' generate input layer of submodel
        Args:
            table: the input dict of submodel
        Return:
            input layer
        '''
        data_input = ""
        for info in table.keys():
            data_input += 'layer {\n'
            data_input += '    name: "' + table[info]['name'] + '"\n'
            data_input += '    type: "Input"\n'
            data_input += '    top: "' + table[info]['name'] + '"\n'
            data_input += '    input_param {\n'
            data_input += '        shape {\n'
            for j in range(len(table[info]['shape'])):
                data_input += '        dim: ' + str(
                    table[info]['shape'][j]) + '\n'
            data_input += '        }\n'
            data_input += '    }\n'
            data_input += '}\n'

        return data_input

    def connect_to_datainput(self, net, net1, input_index, save_layer_lst):
        ''' connect input layers
        '''
        table = {}
        input_layer_lst = []
        save_layer_name_lst = []
        for i in save_layer_lst:
            save_layer_name_lst.append(net.layer[i].name)
        for i in input_index:
            bottom_names = net.layer[i].bottom
            input_layer_table = {}
            if bottom_names:
                for index in range(len(bottom_names)):
                    if bottom_names[index] not in save_layer_name_lst:
                        # 考虑input节点有相连的情况
                        bottom_name_shape = {}
                        if bottom_names[index] not in table:
                            bottom_name_shape['shape'] = net1.blobs[
                                bottom_names[index]].data.shape
                            bottom_name_shape['name'] = "data_" + \
                                                        bottom_names[index]
                            table[bottom_names[index]] = copy.deepcopy(
                                bottom_name_shape)
                            net.layer[i].bottom[index] = "data_" + \
                                                         bottom_names[index]
                        else:
                            net.layer[i].bottom[index] = table[
                                bottom_names[index]]['name']
            else:
                input_layer_table['name'] = net.layer[i].name
                input_layer_table['shape'] = net1.blobs[
                    net.layer[i].name].data.shape
                input_layer_table['dtype'] = net.layer[i].type
                input_layer_lst.append(input_layer_table)
        return table, input_layer_lst

    def tans2inputinfo(self, table):
        ''' generate input dict based on submodel input
        '''
        lst = []
        for info in table.keys():
            input_dic = {}
            input_dic['name'] = table[info]['name']
            input_dic['shape'] = table[info]['shape']
            input_dic['dtype'] = "Input"
            lst.append(input_dic)
        return lst

    def caffe_subnet_extract(self, input_name, output_name, net, net1):
        ''' extract submodel from caffe net
        Args:
            input_name: input layers
            output_name: output layers
            net: caffe net of prototxt
            net1: caffe net of prototxt & caffemodel
        Return:
            data_input_all: input dict of submodel
            final_net: net of prototxt
            final_net2 : net of prototxt & caffemodel
        '''
        import caffe
        input_net = copy.deepcopy(net)
        after_modify_deploy_net = "after_modify_deploy.prototxt"
        before_modify_caffemodel = "before_modify_deploy.caffemodel"
        if input_net and input_net.layers:
            raise RuntimeError("Use the upgrade_net_proto_text and "
                               "upgrade_net_proto_binary tools that "
                               "ship with Caffe to upgrade them first")
        self.check_layername(input_net, output_name)
        self.check_layername(input_net, input_name)
        # Cut the network
        input_index = self.get_layers_index(input_net, input_name)
        layer_output_forward = self.find_layer_forward(input_net, output_name)
        layers_input_forward = []
        for name in input_name:
            layer_input_forward = self.find_layer_forward(input_net, [name])
            names_except = (set(input_index) -
                            set(self.get_layers_index(input_net, [name])))
            if not set(layer_input_forward).intersection(names_except):
                layers_input_forward.extend(list(set(layer_input_forward)))
            else:
                raise RuntimeError(
                    "invalid start nodes, any two of "
                    "the start nodes should not be accessed to each other")

        save_output_forward = (list(set(layer_output_forward).difference(
            set(layers_input_forward))))

        if set(layers_input_forward).issubset(set(layer_output_forward)):
            save_layer_lst = list(
                set(save_output_forward).union(set(input_index)))
            table, input_layer_lst = self.connect_to_datainput(
                input_net, net1, input_index, save_layer_lst)
            data_input = self.get_datainput(table)
            self.del_layer(input_net, save_layer_lst)
            self.save_prototxt(input_net, after_modify_deploy_net, data_input)
            try:
                if self.check_network(after_modify_deploy_net):
                    final_net = self.read_prototxt(after_modify_deploy_net)
                    net1.save(before_modify_caffemodel)
                    final_net2 = caffe.Net(
                        after_modify_deploy_net,
                        before_modify_caffemodel,
                        caffe.TEST)
                    os.remove(after_modify_deploy_net)
                    os.remove(before_modify_caffemodel)
                    data_input_all = self.tans2inputinfo(table)
                    data_input_all.extend(input_layer_lst)
                    return data_input_all, final_net, final_net2
            except RuntimeError as ex:
                raise RuntimeError(ex)
        else:
            raise RuntimeError(
                "Please cheak input and output nodes in user.config.")

    def save_input_data(self, golden_data_path, model_name, data_input):
        ''' save submodel input data
        '''
        lst = []
        for item in data_input:
            data = np.random.normal(0, 1, size=item['shape'])
            save_path = os.path.join(golden_data_path, model_name,
                                     item['name'] + '.bin')
            lst.append(save_path)
            data.tofile(file=save_path, sep="", format="%s")
        save_path = os.path.join(golden_data_path, model_name,
                                 'input_desc.txt')
        with open(save_path, 'w') as desc_file:
            for i in lst:
                desc_file.write(str(i))
                desc_file.write('\n')

    def create_goldden_data(
            self,
            net,
            golden_data_path,
            filename,
            input_info,
            output_node):
        ''' generate input/output golden data
        '''
        input_name_list = []
        input_data_list = {}
        current_input_path = os.path.join(golden_data_path, filename)
        if os.path.exists(current_input_path):
            os.rmdir(current_input_path)
        os.mkdir(current_input_path)

        model_input_info = ""
        for inf in input_info:
            input_data = np.random.uniform(
                1, 100, inf['shape']).astype(
                np.float32)

            if not model_input_info:
                model_input_info += inf['name'] + ":" + \
                    ','.join((str(i) for i in inf['shape']))
            else:
                model_input_info += ";" + inf['name'] + ":" + \
                                    ','.join((str(i) for i in inf['shape']))
            input_name_list.append(inf['name'])
            input_data_list[inf['name']] = input_data
            net.blobs[inf['name']].data[...] = input_data

        input_data_name = os.path.join(
            current_input_path,
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
