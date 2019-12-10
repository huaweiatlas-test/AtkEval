import numpy as np
import os


def data_save(feed_dict_collection, res_dict_collection, golden_data_path):
    '''
    1.save the randomly generated input data and output
    2.the single op model may have more than one input data
    3.current only one output data
    4.All the data are saved as float32 regardless of the acutual data type
    5.ALL the inputs of a model are saved into ONLY ONE binary file,with the
    following format:
    first element: number of inputs of the model
    second element: Byte length of the first input
    third element: Byte length of the second input
    ...
    binary data of the first input
    binary data of the second input
    ...
    6.directory format sample:
        .model_data/model1/input.bin  (may contain multiple inputs)
                             output1.bin
                             input_desc.txt
                             output_desc.txt
        .model_data/model2/input.bin   (may contain multiple inputs)
                             output1.bin
                             output2.bin
                             input_desc.txt
                             output_desc.txt
      the input_desc.txt contains the absolute path of joint input data
      the output_desc.txt contains the absolute path of all the output data
    '''
    input_desc_collection = dict()
    for feed_dict in feed_dict_collection:
        model_spec = os.path.join(golden_data_path, feed_dict['model_name'])
        if not os.path.exists(model_spec):
            os.mkdir(model_spec)
        with open(os.path.join(model_spec, 'input_desc.txt'),
                  'w') as input_desc_file:
            input_desc_file.write(
                os.path.join(os.getcwd(), model_spec, 'input.bin\n'))
        tmp_input_desc = list()
        input_file_array = np.array([len(feed_dict.keys()) - 1
                                     ]).astype(np.float32)
        tmp_model_name = ''
        for name, data in feed_dict.items():
            if name == 'model_name':
                tmp_model_name = data
                continue
            name = name.split(':')[0]
            tmp_input_desc.append(name + ':' +
                                  ','.join(list(map(str, data.shape))))
            input_file_array = np.concatenate(
                (input_file_array, np.array([data.nbytes]).astype(np.float32)),
                axis=0)
        for desc in tmp_input_desc:
            data = feed_dict[desc.split(':')[0] + ':0']
            data = data.astype(np.float32)
            if len(data.shape) == 4:
                data = np.transpose(data, [0, 3, 1, 2])
            # input_file.write(struct.pack('i', data))
            input_file_array = np.concatenate(
                (input_file_array, data.astype(np.float32).flatten()), axis=0)
        input_file_array.tofile(
            os.path.join(os.getcwd(), model_spec, 'input.bin'))
        input_desc_collection[tmp_model_name] = ';'.join(tmp_input_desc)

    for res_dict in res_dict_collection:
        model_spec = os.path.join(golden_data_path, res_dict['model_name'])
        output_desc_file = open(os.path.join(model_spec, 'output_desc.txt'),
                                'w')
        for name, data in res_dict.items():
            if name == 'model_name':
                continue
            name = name.split(':')[0]
            name = name.replace('/', '_')
            data = data.astype(np.float32)
            if len(data.shape) == 4:
                data = np.transpose(data, [0, 3, 1, 2])
            data.tofile(os.path.join(model_spec, name) + '.bin')

            output_desc_file.write(
                os.path.join(os.getcwd(), model_spec, name) + '.bin')
            output_desc_file.write('\n')

    return input_desc_collection
