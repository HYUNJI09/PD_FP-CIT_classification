import os
import copy
import shutil

import pandas as pd

from dict_utils import *



def get_full_subdir(upper_dir):
    lower_dir = os.listdir(upper_dir)
    lower_full_dir = [os.path.join(upper_dir, c) for c in lower_dir]
    return lower_full_dir

def compare_two_set(A_list, B_list):
    #B_list_ = B_list # swallow copy
    B_list_ = copy.deepcopy(B_list)
    print("comparing A and B set")
    A_B_intersection = []
    B_complementary = []
    for elem_a in A_list:
        #elem_a = str(elem_a)
        cond_flag = False
        for ind, elem_b in enumerate(B_list_):
            #elem_b = str(elem_b)
            if elem_a == elem_b:
                cond_flag = True
                # print("in", pid, filename_)
                A_B_intersection.append(elem_a)
                del B_list_[ind]
                break
        if cond_flag is False:
            # print("not in", pid, filename_)
            B_complementary.append(elem_a)

    return B_complementary, B_list_, A_B_intersection # only A set, only B set, intersection

def check_list_set_state(A, B):
    A_, B_, AnB = compare_two_set(A, B)
    print("A", A)
    print("A len", len(A))
    print("B", B)
    print("B len", len(B))
    print("A_", A_)
    print("A_ len", len(A_))
    print("B_", B_)
    print("B_ len", len(B_))
    print("AnB", AnB)
    print("AnB len", len(AnB))

def parser_fbb(filename):
    basename_ = os.path.basename(filename)
    #basename_ = basename.split(".")[0]
    #print(basename_)
    if "_" in basename_:
        return basename_[2:-8]
    elif "x" in basename_:
        return basename_[3:-6]


def create_mapping_dict(reference_path, mapping_info, save_path):
    standard_mapping_filepath = os.path.join(reference_path)

    mapping_df = pd.read_excel(standard_mapping_filepath)
    print(len(mapping_df[mapping_info[0]].tolist()))
    print(len(mapping_df[mapping_info[[1]]].tolist()))

    pid_org = mapping_df[mapping_info[0]].tolist()
    pid_new = mapping_df[mapping_info[1]].tolist()

    org2new_dict = dict()
    new2org_dict = dict()
    for ind, (org, new) in enumerate(zip(pid_org, pid_new)):
        org2new_dict[str(org)] = str(new)
        new2org_dict[str(new)] = str(org)

    save_dict(org2new_dict, os.path.join(save_path, "org2new_dict"))
    save_dict(new2org_dict, os.path.join(save_path, "new2org_dict"))
    return

def create_mapping_dict_from_key_value(key_value_pair, save_dir, save_dict=False):
    org2new_dict = dict()
    new2org_dict = dict()
    for key, value in zip(key_value_pair[0], key_value_pair[1]):
        org2new_dict[str(key)] = str(value)
        new2org_dict[str(value)] = str(key)

    if save_dict:
        save_dict(org2new_dict, os.path.join(save_dir, "org2new_dict"))
        save_dict(new2org_dict, os.path.join(save_dir, "new2org_dict"))

        anonymity_dict = dict()
        anonymity_dict["org"] = key_value_pair[0]
        anonymity_dict["new"] = key_value_pair[1]
        excel_df = pd.DataFrame(anonymity_dict)
        excel_df.to_excel(os.path.join(save_dir, "matching.xlsx"))
    return org2new_dict, new2org_dict


def revise_col_elem_from_excel(excel_path, save_path, col_name, parser):
    excel_df = pd.read_excel(excel_path)
    excel_df[col_name] = [str(parser(elem)) for elem in excel_df[col_name].tolist()]
    excel_df.to_excel(save_path)
    return excel_df
"""
excel_path = "C:\\Users\\NM\\PycharmProjects\\MDBManager\\query_list_needed_rivision.xlsx"
save_path = "C:\\Users\\NM\\PycharmProjects\\MDBManager\\query_list_rivised.xlsx"
col_name = "PatientID"
def parser_(test_str):
    return test_str[:-1]
test = revise_col_elem_from_excel(excel_path, save_path, col_name, parser=parser_)
print(test)
"""

def revise_filename_from_dir(target_dir, parser):
    target_filelist = [os.path.join(target_dir, elem) for elem in os.listdir(target_dir)]
    for elem in target_filelist:
        dirname_ = os.path.dirname(elem)
        basename_ = os.path.basename(elem)

        extention = basename_.split(".")[-1]
        revised = parser(basename_)
        rename_path = os.path.join(dirname_, revised+"."+extention)
        shutil.move(elem, rename_path)
    return
"""
def parser_(test_str):
    print(test_str)
    tmp_ = test_str.split(".")[0]
    pid_ = tmp_[-5:-2]
    print(test_str)
    slice_ = test_str.split(".")[-2]

    new_name = pid_+"_"+slice_
    return new_name
revise_filename_from_dir(target_dir, parser=parser_)
"""

if __name__ == "__main__":
    def parser_(test_str):
        print(test_str)

        return test_str[:-9]
    target_dir = "E:\\export_fbb_ks_university\\190427\\190408_FBB_add_patient_13_all_Anonymize_nii_result2\\cw_de_anonymize"

    revise_filename_from_dir(target_dir, parser=parser_)