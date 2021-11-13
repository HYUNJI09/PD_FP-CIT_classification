import os
import sys
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from ImageDataIO import ImageDataIO

class ExcelDataIO():
    def __init__(self):
        return

    def read_file(self, source_path, query_id = "ID", sheet_name=None, columns=None):
        pd_df = pd.read_excel(source_path, sheet_name=sheet_name)
        #print(sheet_name, "sheet nan check", pd_df.isnull())
        #print(sheet_name, "sheet nan check", pd_df.isnull().any().any())
        # nan value check
        if pd_df.isnull().any().any() :
            print("[!] nan value detected")
            print(sheet_name, "nan rows", pd_df[pd_df.isnull().any(1)])

        # sample id
        sample_id_list = pd_df[query_id].tolist()
        sample_id_list = [self._ensure_str(item) for item in sample_id_list]

        # column id
        #print("column id",pd_df.columns.tolist())
        col_list = pd_df.columns.tolist()
        feature_list = []
        pd_matrix = pd_df.as_matrix()
        for ind in range(len(sample_id_list)):
            if columns is None:
                feature_list.append(pd_df[ind][1:])
            else:
                sample_feature = []
                for col_name in columns:
                    col_ind = col_list.index(col_name)
                    sample_feature.append(pd_matrix[ind][col_ind])
                feature_list.append(sample_feature)
        return feature_list, sample_id_list

    def _ensure_str(self, input_value):
        if isinstance(input_value, str):
            if input_value == "nan":
                return input_value
            elif "." in input_value:
                return str(int(float(input_value)))
        elif isinstance(input_value, int):
            return str(input_value)
        elif isinstance(input_value, float):
            if math.isnan(input_value) :
                return "nan"
            else :
                return str(int(float(input_value)))


if __name__ == "__main__":
    print("hello world")
    # extention = "png"
    # is2D = False
    # view="axial"
    # source_path = "C:\\Users\\NM\\PycharmProjects\\Med_Labeling\\images.png"
    # idio = ImageDataIO(extention, is2D)
    # img = idio.read_file(source_path)
    # img.show()
    # print("size", img.size)
    #



    # source_path = "C:\\Users\\NM\\PycharmProjects\\dicom_AD_test_data\\dicom_AD_test_data\\ct_AD_ID_001_1.2.410.200055.998.998.1707237463.28700.1526446844.649.dcm"
    source_path = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\RCTU_t-SNE\\dataset\\Slice grading_transpose.xlsx"
    idio = ExcelDataIO()
    features, pid = idio.read_file(source_path, query_id = "Slice No", sheet_name="BAPL2")  # img.show()

    print("features", features)
    sample_ind = 22
    print("pid", pid[sample_ind])
    feature_ind = 14
    print("features[0]", features[sample_ind])
    print("len(features[0])", len(features[sample_ind]))
    print("features[sample_ind][feature_ind]", features[sample_ind][feature_ind])
    # print("pid", pid)