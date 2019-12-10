# coding=UTF-8
import argparse
import os
import subprocess
from multiprocessing import Pool
from .common_info import CommonInfo


class Model:
    ''' model info class
    '''

    def __init__(self, model_path, input_info, framework, target):
        self._input = input_info
        self._model = ''
        self._weight = None
        paths = model_path.split('/')
        self.om_name = paths[-1]
        self.om_model = os.path.join(target, self.om_name)
        self._framework = str(framework)
        if framework == 0:
            self._model = model_path + '.prototxt'
            self._weight = model_path + '.caffemodel'
        else:
            self._model = model_path + '.pb'
            if not input_info:
                try:
                    self._input = self.__get_tf_input(self._model)
                except Exception:
                    print("Could not get input shape.")
        self.command = self.__gen_omg_command()

    def __get_tf_input(self, pb_path):
        ''' get input shape of tensorflow model
        '''
        result = ''
        input_dim = ''
        with open(pb_path, 'rb') as pb_file:
            import tensorflow as tf
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(pb_file.read())
            for node in graph_def.node:
                if node.op == 'Placeholder':
                    name = getattr(node, 'name')
                    dims = getattr(node.attr['shape'].shape, 'dim')
                    if len(dims) > 4:
                        print("Unsupport input shape.")
                        return ''
                    # CNN model for 4-input && input format is NHWC
                    if len(dims) == 4:
                        if getattr(dims[0], 'size') == -1:
                            input_dim = '1,'
                        else:
                            input_dim = str(getattr(dims[0], 'size')) + ','
                        if getattr(dims[1], 'size') == -1 and \
                                getattr(dims[2], 'size') == -1:
                            input_dim += '224,224,'
                        else:
                            input_dim += str(getattr(dims[1], 'size')) + ',' \
                                + str(getattr(dims[2], 'size')) + ','
                        input_dim += str(getattr(dims[3], 'size'))
                    elif len(dims) == 3:
                        if getattr(dims[0], 'size') == -1:
                            input_dim = '1,'
                        else:
                            input_dim = str(getattr(dims[0], 'size')) + ','
                        if getattr(dims[1], 'size') == -1 and \
                                getattr(dims[2], 'size') == -1:
                            input_dim += '224,224'
                        else:
                            input_dim += str(getattr(dims[1], 'size')) + ',' \
                                + str(getattr(dims[2], 'size'))
                    # NLP model
                    elif len(dims) == 2:
                        if getattr(dims[0], 'size') == -1:
                            input_dim = '1,'
                        else:
                            input_dim = str(getattr(dims[0], 'size')) + ','

                        if getattr(dims[1], 'size') == -1:
                            input_dim += '128'
                        else:
                            input_dim += str(getattr(dims[1], 'size'))
                    elif len(dims) == 1:
                        if getattr(dims[0], 'size') == -1:
                            input_dim = '1,'
                        else:
                            input_dim = str(getattr(dims[0], 'size'))
                    elif getattr(node.attr['shape'].shape, 'unknown_rank'):
                        input_dim = '1,224,224,3'
                        print('Input shape is unknown_rank. \
                        Please convert manually.')
                    # use ';' to split different inputs
                    if result != '':
                        result += ';'
                    if input_dim != '':
                        result += name + ':' + input_dim
        return result

    def __gen_omg_command(self):
        ''' generate omg convert command
        '''
        model = ' --model=' + self._model
        weight = ' --weight=' + self._weight if self._weight else ''
        output = ' --output=' + self.om_model
        framework = ' --framework=' + self._framework
        input_str = ' --input_shape="' + self._input + '"' if self._input \
            else ''
        check_file = ' --check_report=' + self.om_model + '.json'
        log = ' > convert.log 2>&1 '
        command = model + weight + output + framework + input_str + \
            check_file + log
        return command


class Converter:
    ''' converter class: convert models to D-models
    '''

    def __init__(self):
        self.model_list = set()
        self.success_list = set()
        self.print_log = True

    def turn_on_convert_log(self, is_print_log):
        ''' set print log attribute
        '''
        self.print_log = is_print_log

    def is_print_convert_log(self):
        ''' get print log attribute
        '''
        return self.print_log

    def search_model_list(self, models_dir, model_type):
        ''' get model list from models_dir
        Args:
            models_dir: the folder path of models
            model_type: framework (caffe or tensorflow)
        '''
        if model_type == 0:
            prototxt = set()
            caffemodel = set()
            self.__search_file(models_dir, '.prototxt', prototxt)
            self.__search_file(models_dir, '.caffemodel', caffemodel)
            self.model_list = prototxt & caffemodel
        elif model_type == 3:
            self.__search_file(models_dir, '.pb', self.model_list)

    def parse_model_list(self, input_dict):
        ''' parse model input dict to get model info list
        Args:
            input_dict: dict of model input. {model:input_shape_str}
        '''
        sub_model_path = CommonInfo.get_sub_model_path()
        for model in input_dict:
            self.model_list.add(os.path.join(sub_model_path, model))

    def __search_file(self, root, model_ext, file_list):
        ''' search model file from root dir
        '''
        items = os.listdir(root)
        for item in items:
            path = os.path.join(root, item)
            if os.path.isdir(path):
                self.__search_file(path, model_ext, file_list)
            else:
                name, ext = os.path.splitext(path)
                if ext == model_ext:
                    file_list.add(name)

    def parse_res(self, resDict):
        ''' parse converter result
        '''
        if resDict['res'] == 0:
            self.success_list.add(resDict['name'])
        if self.print_log:
            if resDict['res'] == 0:
                print('convert {} success'.format(resDict['name'] + '.om'))
            else:
                print('convert {} fail'.format(resDict['name']))


def call_omg(OMG, model_file, input_info, model_type, result_dir):
    ''' call omg to convert model
    '''
    model = Model(model_file, input_info, model_type, result_dir)
    ret = subprocess.call(OMG + model.command, shell=True)
    remove_file(model.om_model + '.json')
    return {'name': model.om_name, 'res': ret}


def remove_file(path):
    ''' remove file
    '''
    if os.path.isfile(path):
        os.remove(path)


def verify_file_path(path):
    ''' check validity of file
    '''
    if not os.path.isfile(path):
        print("{} does not exist.".format(path))
        return -1
    return 0


def verify_dir_path(path):
    ''' check validity of directory
    '''
    if not os.path.isdir(path):
        print("{} is not a directory! Please check.".format(path))
        return -1
    return 0


def verify_framework(convert_type):
    ''' check validity of convert type
    '''
    if convert_type != 'caffe' and convert_type != 'tensorflow':
        print("{} is unsupported. please check.".format(convert_type))
        return -1
    return 0


def search_omg(ddk_path):
    ''' search omg
    '''
    gcc_item = ''
    file_names = os.listdir(os.path.join(ddk_path, 'lib'))
    for name in file_names:
        if name.startswith('x86_64'):
            gcc_item = name
    os.environ['LD_LIBRARY_PATH'] = os.path.join(ddk_path, 'lib', gcc_item)
    omg_path = os.path.join(ddk_path, 'bin', gcc_item, 'omg')
    return omg_path


def convert_model(ddk,
                  models,
                  convert_type,
                  result,
                  input_dict=None,
                  print_log=True):
    ''' convert model to D
    '''
    convert_type = convert_type.strip().lower()
    if verify_dir_path(ddk) == -1 or verify_dir_path(models) == -1 \
            or verify_framework(convert_type) == -1:
        return None
    ddk_path = os.path.abspath(ddk)
    models_dir = os.path.abspath(models)
    result_dir = os.path.abspath(result)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    framework = 0 if convert_type == 'caffe' else 3
    converter = Converter()
    converter.turn_on_convert_log(print_log)
    if not input_dict:
        converter.search_model_list(models_dir, framework)
    else:
        converter.parse_model_list(input_dict)
    OMG = search_omg(ddk_path)
    if verify_file_path(OMG) == -1:
        print("Please check DDK path.")
        return None

    if converter.is_print_convert_log():
        print('Convert model...... Please wait')
    pool = Pool(8)
    for model in converter.model_list:
        input_str = None
        if input_dict:
            name = model.split('/')[-1]
            input_str = input_dict.get(name)
            if not input_str:
                print('Input of {} is invalid. Please check.'.format(name))
                continue
        pool.apply_async(call_omg,
                         args=(OMG, model, input_str, framework, result_dir),
                         callback=converter.parse_res)
    pool.close()
    pool.join()
    remove_file('convert.log')
    if converter.is_print_convert_log():
        print('convert finish!')
    return converter.success_list


def init_args():
    ''' parse input args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--ddk_home',
                        type=str,
                        default='',
                        help='ddk path')
    parser.add_argument('-t',
                        '--type',
                        type=str,
                        default='tensorflow',
                        help='input caffe or tensorflow')
    parser.add_argument('-m',
                        '--models_dir',
                        type=str,
                        default='',
                        help='models')
    parser.add_argument('-o', '--output', type=str, default='', help='output')
    return parser.parse_args()


if __name__ == '__main__':
    '''
    python convert2davinci.py -d=/home/mindstudio/tools/che/ddk/ddk/ \
    -m=../model/ -t=caffe -o=./result/
    '''
    args = init_args()
    convert_model(args.ddk_home, args.models_dir, args.type, args.output)
