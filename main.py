import configparser
import os
import shutil
from caffe_impl.caffe_compare import CaffeCompare
from tensorflow.python.util import deprecation
from tf_impl.tf_basic_compare import TensorflowBasicCompare
from utils.convert2davinci import convert_model

deprecation._PRINT_DEPRECATION_WARNINGS = False


def parameter_check(config):
    '''check the params in user.config'''
    common_key = ['ddk', 'framework', 'mode', 'result', 'model']
    caffe_key = ['caffe', 'weight']
    subnet_key = ['start_nodes', 'end_nodes']
    for key in common_key:
        if key not in list(config.keys()):
            print("{key} not in use config, \
            please check it by yourself.".format(key=key))
            return 1

    if 'name' in list(config['framework']):
        framework = config['framework']['name']
        if framework not in ['caffe', 'tensorflow']:
            print("the value of name must be caffe or tensorflow, now is :\
                {name}, please check it by yourself".format(name=framework))
            return 1

        if framework == "caffe":
            for key in caffe_key:
                if key not in list(config.keys()):
                    print("key {key} is not in user config, \
                    please check it by yourself.".format(key=key))
                    return 1
                if 'path' in config[key]:
                    path = config[key]['path']
                    if not os.access(path, os.R_OK):
                        print("There is no right to read {path}, \
                        please check it by yourself.".format(path=path))
                        return 1
                else:
                    print("the key of path is not in {key}, \
                    please check it by yourself.".format(key=key))
                    return 1
    else:
        print(
            "the key of name is not in framework please check it by yourself.")

    if 'number' in list(config['mode']):
        number = config['mode']['number']
        if str(number) not in ['1', '2', '3']:
            print("The mode must be in 1,2,3, now is:\
            {mode}".format(mode=number))
            return 1

        if str(number) == '2':
            for key in subnet_key:
                if key not in list(config.keys()) or 'name' not in list(
                        config[key]):
                    print('{key} not in user config or key name not in {key},\
                     please check it by yourself.'.format(key=key))
                    return 1
    else:
        print('The key of number is not in mode, \
        please check it by yourself.')
        return 1

    if 'path' in list(config['ddk']):
        path = config['ddk']['path']
        if not os.access(path, os.R_OK):
            print("There is no right to read {path}".format(path=path))
            return 1
    else:
        print("The key of path is not in ddk, please check it by yourself.")
        return 1

    if 'path' not in list(config['result']):
        print("The key of path is not in result, please check it by yourself")
        return 1

    if 'path' in list(config['model']):
        path = config['model']['path']
        if not os.access(path, os.R_OK | os.W_OK):
            print("There is no right to read {path}, \
            please check it by yourself".format(path=path))
            return 1
    else:
        print("The key of path is not in model, please check it by yourself")
        return 1
    return 0


def config_parser(config_path):
    '''
    read the user_config file and return a config info dict
    :param config_path: the path of the config file
    :return: config info dict
    '''
    if not os.path.exists(config_path):
        print(config_path + ' not exists, exit!')
        exit()
    config = configparser.ConfigParser()
    config.read(config_path)
    result = parameter_check(config)
    if result == 1:
        exit()
    user_config_dict_ = dict()
    user_config_dict_['ddk_path'] = config['ddk']['path']
    user_config_dict_['model_path'] = config['model']['path']
    user_config_dict_['framework_name'] = config['framework']['name']
    user_config_dict_['result_path'] = config['result']['path']
    user_config_dict_['mode_number'] = config['mode']['number']

    if user_config_dict_['framework_name'] == 'caffe':
        user_config_dict_['caffe_path'] = config['caffe']['path']
        user_config_dict_['weight_path'] = config['weight']['path']

    if int(user_config_dict_['mode_number']) == 2:
        user_config_dict_['start_nodes'] = config['start_nodes'][
            'name'].replace(' ', '').split(',')
        user_config_dict_['end_nodes'] = config['end_nodes']['name'].replace(
            ' ', '').split(',')

    return user_config_dict_


def rename_weight(root):
    '''rename the name of weights'''
    items = os.listdir(root)
    weight_new_name = None
    weight_old_name = None
    for item in items:
        path = os.path.join(root, item)
        name, ext = os.path.splitext(path)
        if ext == '.prototxt':
            weight_new_name = name + '.caffemodel'
        elif ext == '.caffemodel':
            weight_old_name = path
    os.rename(weight_old_name, weight_new_name)


def model_check(user_config_dict):
    '''
    check the validation of caffe model using omg
    :param user_config_dict: config info dict
    :return:
    '''
    print('Checking model validation...')
    tmp_floder = os.path.abspath('.model_check_tmp')
    framewrok = user_config_dict['framework_name'].lower()
    if not os.path.isfile(user_config_dict['model_path']):
        raise RuntimeError('prototxt file not exists, please check!')
    if framewrok == 'caffe' and (not os.path.isfile(
            user_config_dict['weight_path'])):
        raise RuntimeError('caffemodel file not exists, please check!')
    if os.path.exists(tmp_floder):
        shutil.rmtree(tmp_floder)
    os.mkdir(tmp_floder)
    shutil.copy(user_config_dict['model_path'], tmp_floder)
    if framewrok == 'caffe':
        shutil.copy(user_config_dict['weight_path'], tmp_floder)
        rename_weight(tmp_floder)
    ret = convert_model(user_config_dict['ddk_path'],
                        tmp_floder,
                        framewrok,
                        tmp_floder,
                        print_log=False)
    shutil.rmtree(tmp_floder)

    if ret:
        return 1
    else:
        return 0


if __name__ == '__main__':
    user_config_dict = config_parser('user.config')

    if not model_check(user_config_dict):
        print('model check failed, please check!')
        exit()

    if user_config_dict['framework_name'].lower() == 'tensorflow':
        tf_compare_instance = TensorflowBasicCompare(user_config_dict)
        tf_compare_instance.run()
    elif user_config_dict['framework_name'].lower() == 'caffe':
        caffe_compare_instance = CaffeCompare(user_config_dict)
        caffe_compare_instance.run()
    else:
        raise RuntimeError('unsupported framework name')
