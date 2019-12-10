import numpy as np
import os


def name_split(file_name):
    '''split out the file name'''
    return file_name[17:-6]


def precision_cal(davinci_data_array, tf_data_array):
    '''calculate precision with respect to different metrics'''
    # Euclidean Distance
    euc_distance = np.linalg.norm(davinci_data_array -
                                  tf_data_array) / davinci_data_array.shape[0]

    # relative distance
    rel_distance = np.mean(
        np.abs(davinci_data_array - tf_data_array) /
        (np.abs(tf_data_array) + 1e-6))

    # cosine distance
    cos_distance = np.dot(davinci_data_array, tf_data_array) / (
        np.linalg.norm(davinci_data_array) * np.linalg.norm(tf_data_array))

    return euc_distance, rel_distance, cos_distance


def str_pad(str_to_pad, pad_len):
    '''
    pad th str to the specific length for more readable display
    '''

    return '{:<{len}}'.format(str_to_pad, len=pad_len)


def precision_compare(davinci_data_path, golden_data_path,
                      valid_model_name_list, result_path):
    '''
    compare the results between davinci results and tf results, print and write
    the results to file
    1. find one result in davinci_result_path
    2. find the matched result in tf results
    3. compare
    4. print and save
    '''
    compare_result_dict = dict()
    max_name_len = 0

    # for specific or dichotomy comparing mode, there is only one model called
    # sub_graph, it may have more than 1 outputs
    if len(valid_model_name_list
           ) == 1 and valid_model_name_list[0] == 'sub_graph':
        golden_output_desc_path = os.path.join(golden_data_path,
                                               valid_model_name_list[0],
                                               'output_desc.txt')

        davinci_data_files = os.listdir(davinci_data_path)
        for file_handle in davinci_data_files:
            if not (file_handle.startswith('sample')
                    and 'escaped_time' not in file_handle):
                continue

            with open(golden_output_desc_path, 'r') as golden_output_file:
                golden_output_path_list = golden_output_file.readlines()
            for golden_output_path in golden_output_path_list:
                golden_output_path = golden_output_path.strip()
                golden_output_name = golden_output_path.split('/')[-1][0:-4]
                # find the mapped output
                if golden_output_name == name_split(file_handle):

                    # read the davinci data
                    davinci_data_array = np.fromfile(
                        os.path.join(davinci_data_path, file_handle),
                        np.float32)

                    # read the tf/caffe data in binary
                    golden_data_array = np.fromfile(golden_output_path,
                                                    np.float32)

                    # check the length
                    if davinci_data_array.shape[0] != golden_data_array.shape[
                            0]:
                        print(
                            'FAILED to compare, the length of davinci and '
                            'tf/caffe results are NOT identical, please check!'
                        )
                        continue

                    # compare
                    euc_distance, rel_distance, cos_distance = precision_cal(
                        davinci_data_array, golden_data_array)
                    compare_result_dict[golden_output_name] = [
                        euc_distance, rel_distance, cos_distance
                    ]

                    # update max_name_len for better display
                    if len(golden_output_name) > max_name_len:
                        max_name_len = len(golden_output_name)

    else:
        # for node by node comparing mode, it is more efficient to iterate \
        # the valid_model_name_list
        for model_name in valid_model_name_list:
            golden_output_desc_path = os.path.join(golden_data_path,
                                                   model_name,
                                                   'output_desc.txt')
            if not os.path.exists(golden_output_desc_path):
                continue
            with open(golden_output_desc_path, 'r') as golden_output_file:
                golden_output_path_list = golden_output_file.readlines()
            for golden_output_path in golden_output_path_list:
                golden_output_path = golden_output_path.strip()
                golden_output_name = golden_output_path.split('/')[-1][0:-4]
                # for node by node comparing mode, we assume every model only
                # have 1 output, some ops like 'Split' are
                # NOT supported
                davinci_output_path = os.path.join(
                    davinci_data_path,
                    'sample1_output_0_' + golden_output_name + '_0.bin')
                if os.path.exists(davinci_output_path) and os.path.exists(
                        golden_output_path):

                    # read the davinci data
                    davinci_data_array = np.fromfile(
                        os.path.join(davinci_output_path), np.float32)

                    # read the tf/caffe data in binary
                    golden_data_array = np.fromfile(golden_output_path,
                                                    np.float32)

                    # check the length
                    if davinci_data_array.shape[0] != golden_data_array.shape[
                            0]:
                        print(
                            'FAILED to compare, the length of davinci and'
                            ' tf/caffe results are NOT identical, please check!'
                        )
                        continue

                    # compare
                    euc_distance, rel_distance, cos_distance = precision_cal(
                        davinci_data_array, golden_data_array)
                    compare_result_dict[golden_output_name] = [
                        euc_distance, rel_distance, cos_distance
                    ]

                    # update max_name_len for better display
                    if len(golden_output_name) > max_name_len:
                        max_name_len = len(golden_output_name)

    # print and save
    compare_result_file = open(result_path + '/compare_result_file.txt', 'w')
    property_line = str_pad('Node_name', max_name_len + 5) + str_pad(
        'euc_distance', 20) + str_pad('rel_distance', 20) + str_pad(
            'cos_distance', 20)
    compare_result_file.write(property_line + '\n')
    print('##############compare result####################')
    print(property_line)
    for key, value in compare_result_dict.items():
        node_str = str_pad(key, max_name_len + 5)
        ed_str = str_pad(str(round(value[0], 6)), 20)
        rd_str = str_pad(str(round(value[1], 6)), 20)
        cd_str = str_pad(str(round(value[2], 6)), 20)
        print(node_str + ed_str + rd_str + cd_str)
        compare_result_file.write(node_str + ed_str + rd_str + cd_str + '\n')
    compare_result_file.close()

    return compare_result_dict
