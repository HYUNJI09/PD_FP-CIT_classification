import os
import io
import json

import numpy as np
import pandas as pd
from custom_exception import CustomException

def save_json(dictionary, save_path):
    json_str = json.dumps(dictionary)

    with open(save_path, "w") as fw:
        fw.write(json_str)

def load_json(save_path):
    with open(save_path, "r") as fw:
        _json = fw.read()
    return json.load(io.StringIO(_json))

class Logger():
    def __init__(self, log_save_path):
        self.board = dict()
        self.log_save_path = log_save_path

        return

    # call for one by one info
    def record(self, **infos):
        if len(self.board) == 0 :
            for k, v in infos.items():
                self.board[k] = [v]
        else :
            for k, v in infos.items():
                self.board[k].append(v)
        self.log_msg(infos)
        return

    def _get_var_type(self, var):
        if isinstance(var, int) or isinstance(var, np.int32) or isinstance(var, np.int64):
            return 'd'
        elif isinstance(var, float) or isinstance(var, np.float32) or isinstance(var, np.float64):
            return 'f'
        elif isinstance(var, str):
            return 's'
        else:
            print("debug, type(var)", type(var))
            raise CustomException("[!] type of the received var is not defined!")

    def _create_strfmt(self, size_gap, len_var, var_list, align_l=True):
        str_fmt = "" # '%-10s%-10s%-10s\n'

        for ind in range(len_var):
            str_fmt+='%'
            if align_l:
                str_fmt+='-'
            type_str = self._get_var_type(var_list[ind])
            str_fmt+=str(size_gap) + type_str

        str_fmt+='\n'

        return str_fmt

    # save current state of board
    def log_msg(self, **infos):

        if len(self.board)==0:
            field_to_write = sorted(infos.items(), key=lambda x: x[0])
            #print("field_to_write", field_to_write)

            for k, v in field_to_write:
                self.board[k] = []
                self.board[k].append(v)
            with open(os.path.join(self.log_save_path, 'log.txt'), 'a') as f:
                var_list = tuple(map(lambda x: x[0], field_to_write))
                #print("var_list", var_list)
                strFormat = self._create_strfmt(size_gap=20, len_var=len(infos), var_list=var_list)
                strOut = strFormat % var_list
                f.writelines(strOut)

                var_list = tuple(map(lambda x: x[1], field_to_write))
                strFormat = self._create_strfmt(size_gap=20, len_var=len(infos), var_list=var_list)
                strOut = strFormat % var_list
                f.writelines(strOut)



        else :
            field_to_write = sorted(infos.items(), key=lambda x:x[0])
            for k, v in field_to_write:
                 self.board[k].append(v)
            with open(os.path.join(self.log_save_path, 'log.txt'), 'a') as f:
                var_list = tuple(map(lambda x: x[1], field_to_write))
                #print("var_list", var_list)
                strFormat = self._create_strfmt(size_gap=20, len_var=len(infos), var_list=var_list)
                strOut = strFormat % var_list
                f.writelines(strOut)


    def load_board(self):
        self.board = load_json(self.log_save_path)

    def save_board_to_json(self):
        save_json(self.board, save_path=os.path.join(self.log_save_path, "log.json"))

    def save_board_to_excel(self):
        pd_df = pd.DataFrame(self.board)
        pd_df.to_excel(os.path.join(self.log_save_path, "log.xlsx"))