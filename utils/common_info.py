import os


class CommonInfo(object):
    '''class for temporary path info'''
    __base = '.tmp'
    __sub_model = '.tmp/sub_model'
    __golden_data = '.tmp/golden_data'
    __davinci_model = '.tmp/davinci_model'
    __davinci_data = '.tmp/davinci_data'
    __tf_black_list = {'Identity', 'Const', 'Reshape', 'Shape'}
    __caffe_black_list = {'Reshape'}

    @classmethod
    def get_tmp_base_path(cls):
        '''get the top temporary path'''
        return os.path.abspath(cls.__base)

    @classmethod
    def get_sub_model_path(cls):
        '''get the temporary caffe or tensorflow model path'''
        return os.path.abspath(cls.__sub_model)

    @classmethod
    def get_golden_data_path(cls):
        '''get the temporary golden data path'''
        return os.path.abspath(cls.__golden_data)

    @classmethod
    def get_davinci_model_path(cls):
        '''get the temporary davinci model path'''
        return os.path.abspath(cls.__davinci_model)

    @classmethod
    def get_davinci_data_path(cls):
        '''get the temporary davinci infer output data path'''
        return os.path.abspath(cls.__davinci_data)

    @classmethod
    def get_tf_black_list(cls):
        '''get the tensorflow ops which will be ignored'''
        return cls.__tf_black_list

    @classmethod
    def get_caffe_black_list(cls):
        '''get the caffe ops which will be ignored'''
        return cls.__caffe_black_list

    @classmethod
    def is_skip_tf_op(cls, op):
        '''skip the op in tensorflow black list'''
        if op in cls.__tf_black_list:
            return True
        return False

    @classmethod
    def is_skip_caffe_layer(cls, layer):
        '''skip the op in caffe black list'''
        if layer in cls.__caffe_black_list:
            return True
        return False
