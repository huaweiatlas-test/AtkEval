import os
from google.protobuf import text_format


class CaffeUtil(object):
    @classmethod
    def read_prototxt(cls, proto, txt_path):
        ''' get caffenet from prototxt file
        '''
        with open(txt_path, 'rb') as model_file:
            text_format.Parse(model_file.read(), proto)
        return proto

    @classmethod
    def save_prototxt(cls, proto, save_path):
        ''' save caffenet to prototxt
        '''
        with open(save_path, 'w') as model_file:
            model_file.write(text_format.MessageToString(proto))

    @classmethod
    def remove_file(cls, path):
        ''' remove file
        '''
        if os.path.isfile(path):
            os.remove(path)

    @classmethod
    def update_graph(cls, net):
        ''' rename blobs.
        --------------
        Original layer:
        layer{name:layer1, top:top_11, top:top_12}
        layer{name:layer2, bottom:top_11, top:top_21}
        --------------
        Updated layer:
        layer{name:layer1, top:layer1, top:layer1_1}
        layer{name:layer2, bottom:layer1, top:layer2}
        --------------
        '''
        blob_pair = dict()  # blob_pair ->{blob_old_name : blob_new_name}
        for input_name in net.input:
            blob_pair[input_name] = input_name

        layer_index = 0
        for layer in net.layer:
            index = 0
            for old_bottom_name in layer.bottom:
                new_bottom_name = blob_pair.get(old_bottom_name)
                if not new_bottom_name:
                    print('[Error] the bottom is not in blob_pair. please \
                    check this model')
                    print('layer:{},  invalid bottom:{}'.format(
                        layer.name, old_bottom_name))
                    return
                net.layer[layer_index].bottom[index] = new_bottom_name
                index += 1

            index = 0
            for old_top_name in layer.top:
                if index == 0:
                    new_top_name = layer.name
                else:
                    new_top_name = layer.name + '_' + str(index)
                blob_pair[old_top_name] = new_top_name
                net.layer[layer_index].top[index] = new_top_name
                index += 1

            layer_index += 1
