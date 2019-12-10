import os
import subprocess


def davinci_run(davinci_model_path, golden_data_path, davinci_data_path,
                model_name_lsit):
    '''
    run the davinci model
    1.check the existence of input_desc.txt and om model
    2.modify the config.ini and graph_rawdata.prototxt dynamically
    3.run

    :return: a list containing all the valid model names
    '''
    valid_model_name_list = []

    davinci_infer_base_path = os.path.join(davinci_model_path, '../..',
                                           'utils', 'davinci_infer')
    config_ini_path = os.path.join(davinci_infer_base_path, 'config.ini')
    graph_prototxt_path = os.path.join(
        davinci_infer_base_path,
        'test_data/config/graph_rawdata_mutil.prototxt')
    ai_engine_so_path = os.path.join(davinci_infer_base_path,
                                     'libai_engine.so')
    ai_engine_so_path = os.path.abspath(ai_engine_so_path)

    # modify engine_config_path in config.ini, \
    # it will not be changed in the iteration
    with open(config_ini_path, 'r') as file_handle:
        config_content = file_handle.readlines()
    engine_config_path = config_content[4].split(' = ')
    engine_config_path[1] = graph_prototxt_path + '\n'
    config_content[4] = ' = '.join(engine_config_path)
    with open(config_ini_path, 'w') as file_handle:
        file_handle.writelines(config_content)

    # run the om model iteratively
    for model_name in model_name_lsit:
        # currently not supports multi-input model
        om_file_path = os.path.join(os.getcwd(), davinci_model_path,
                                    model_name + '.om')
        input_desc_file_path = os.path.join(golden_data_path, model_name,
                                            'input_desc.txt')
        if os.path.exists(om_file_path) and os.path.exists(
                input_desc_file_path):
            # modify the config.ini
            with open(config_ini_path, 'r') as file_handle:
                config_content = file_handle.readlines()
            test_img_list_path = config_content[3].split(' = ')
            test_img_list_path[1] = input_desc_file_path + '\n'
            config_content[3] = ' = '.join(test_img_list_path)

            result_file_path = config_content[5].split(' = ')
            result_file_path[1] = davinci_data_path + '\n'
            config_content[5] = ' = '.join(result_file_path)

            with open(config_ini_path, 'w') as file_handle:
                file_handle.writelines(config_content)

            # modify the graph_rawdata.prototxt
            with open(graph_prototxt_path, 'r') as file_handle:
                graph_rawdata_content = file_handle.readlines()
            set_om_path = False
            set_so_path = False
            for idx, line in enumerate(graph_rawdata_content):
                # there is only 1 om path in the config, find the om path line
                line = line.strip()
                if line.startswith('value') and 'om' in line:
                    graph_rawdata_content[
                        idx] = 'value:"' + om_file_path + '"\n'
                    set_om_path = True
                elif line.startswith('so_name') and 'libai_engine.so' in line:
                    graph_rawdata_content[
                        idx] = 'so_name:"' + ai_engine_so_path + '"\n'
                    set_so_path = True
                if set_so_path and set_om_path:
                    break
            with open(graph_prototxt_path, 'w') as file_handle:
                file_handle.writelines(graph_rawdata_content)

            cmd = './utils/davinci_infer/DavinciInfer 0 \
            ./utils/davinci_infer/config.ini'

            ret = subprocess.call(cmd, shell=True)
            if ret == 0:
                print(model_name + ' run on Atlas SUCCESS')
                valid_model_name_list.append(model_name)
            else:
                print(model_name + ' run on Atlas FAILED')
    return valid_model_name_list
