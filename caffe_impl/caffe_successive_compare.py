import os
import numpy as np
from . import caffe_util


class SuccesiveCompare(object):
    ''' caffe single layer compare class
    Args:
        proto: empty proto of caffe_pb2.NetParameter()
        txt_path: the path of prototxt file
        bin_path: the path of caffemodel file
    '''

    def __init__(self, proto, txt_path, bin_path):
        self.complete_net = None
        self.model = caffe_util.CaffeUtil.read_prototxt(proto, txt_path)
        self.model_tmp_path = './new_tmp_model.prototxt'
        self.weight_path = bin_path
        self.model_input = set()

    def get_model_info(self):
        ''' generate model input
        '''
        self.model_input.clear()
        for input_name in self.model.input:
            self.model_input.add(input_name)

        for layer in self.model.layer:
            if layer.type == 'Input':
                self.model_input.add(layer.name)

    def save_single_layer(self, sub_model_path):
        ''' save every layer as single-layer model
        Args:
            sub_model_path: the path of saved file
        '''
        import caffe
        from caffe.proto import caffe_pb2
        model_input_dict = dict()
        for layer in self.model.layer:
            if not layer.bottom:
                continue
            save_model_file = os.path.join(sub_model_path,
                                           layer.name + '.prototxt')
            save_weight_file = os.path.join(sub_model_path,
                                            layer.name + '.caffemodel')
            # Do not change any parameter in self.model
            proto = caffe_pb2.NetParameter()
            index = 0
            input_info = list()
            for bottom in layer.bottom:
                new_name = 'top_' + str(index)
                new_layer = proto.layer.add()
                new_layer.name = new_name
                new_layer.top.append(bottom)
                new_layer.type = 'Input'
                input_dims = list()
                blob_shape = caffe_pb2.BlobShape()
                for dim in self.complete_net.blobs[bottom].data.shape:
                    input_dims.append(dim)
                    blob_shape.dim.append(dim)
                new_layer.input_param.shape.append(blob_shape)
                index += 1

                info_dict = dict()
                info_dict['input_name'] = 'top_' + str(index)
                info_dict['blob_name'] = bottom
                info_dict['input_dims'] = input_dims
                input_info.append(info_dict)

            proto.layer.append(layer)
            model_input_dict[layer.name] = input_info

            caffe_util.CaffeUtil.save_prototxt(proto, save_model_file)
            net = caffe.Net(save_model_file, self.weight_path, caffe.TEST)
            net.save(save_weight_file)
        return model_input_dict

    def get_models_input_dict(self, model_dict):
        ''' generate the string of input shape of single-layer models for omg
        Args:
            model_dict: the dict of single-layers models
        '''
        input_str_dict = dict()
        for model_name in model_dict:
            input_str = ''
            info_list = model_dict[model_name]
            for info in info_list:
                if input_str != '':
                    input_str += ';'
                input_str += info['input_name'] + ':'
                for dim in info['input_dims']:
                    input_str += str(dim) + ','
            input_str_dict[model_name] = input_str[:-1]
        return input_str_dict

    def caffe_predictor(self):
        '''' caffe net forward
        '''
        import caffe
        self.complete_net = caffe.Net(self.model_tmp_path, self.weight_path,
                                      caffe.TEST)
        for name in self.model_input:
            input_data = np.random.uniform(1, 100, self.complete_net.blobs[
                name].data.shape).astype(np.float32)
            self.complete_net.blobs[name].data[...] = input_data
        self.complete_net.forward()

    def gen_golden_data(self, save_path, model_dict):
        ''' save data of caffe blob
        Args:
            save_path: the path of save data
            model_dict: the dict of {model_name : input_info}
        '''
        for layer in self.model.layer:
            input_list = model_dict.get(layer.name)
            if not input_list:
                continue

            sub_path = os.path.join(save_path, layer.name)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            input_num = len(input_list)
            input_array = np.array([input_num]).astype(np.float32)

            for info in input_list:
                data_bytes = 4
                for dim in info['input_dims']:
                    data_bytes *= dim
                input_array = np.concatenate(
                    (input_array, np.array(
                        [data_bytes]).astype(
                        np.float32)), axis=0)

            for info in input_list:
                data = self.complete_net.blobs[info['blob_name']].data[...]
                input_array = np.concatenate(
                    (input_array, data.astype(np.float32).flatten()), axis=0)

            input_data_file = os.path.join(sub_path, 'input.bin')
            input_array.tofile(input_data_file)
            with open(os.path.join(sub_path, 'input_desc.txt'), 'w') as desc:
                desc.write(input_data_file + '\n')

            output_data_file_list = list()
            # the name of first top_blob is the same as layer name
            for top_blob in layer.top:
                output_data_file = os.path.join(sub_path, top_blob + '.bin')
                output_data_file_list.append(output_data_file)
                blob_data = self.complete_net.blobs[top_blob].data[...]
                with open(output_data_file, 'wb') as data_file:
                    data_file.write(blob_data.tobytes())

            with open(os.path.join(sub_path, 'output_desc.txt'), 'w') as desc:
                for file_name in output_data_file_list:
                    desc.write(file_name + '\n')

    def successive_compare(self, golden_data_path, sub_model_path):
        ''' caffe single layer compare
        Args:
            golden_data_path: the path of golden data
            sub_model_path: the path of single-layer models
        '''
        caffe_util.CaffeUtil.update_graph(self.model)
        self.get_model_info()
        caffe_util.CaffeUtil.save_prototxt(self.model, self.model_tmp_path)
        self.caffe_predictor()
        single_model_dict = self.save_single_layer(sub_model_path)
        self.gen_golden_data(golden_data_path, single_model_dict)
        input_str_dict = self.get_models_input_dict(single_model_dict)
        caffe_util.CaffeUtil.remove_file(self.model_tmp_path)
        return input_str_dict
