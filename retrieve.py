import os
import glob
import math
import shutil
from collections import Counter

import pydicom

import numpy as np
import pandas as pd

from data_utils import *

#--------------------------------------------------------------------------------------
# related excel list
def make_query_list_from_excel(query_excel_path, query_excel_id, parser=None):
    query_df = pd.read_excel(query_excel_path)
    query_list = [str(elem) for elem in query_df[query_excel_id].tolist()]

    if parser:
        query_list = [str(parser(query_item)) for query_item in query_list]
    return query_list

# retrieve col
def make_col_list_from_excel(query_filename, col_list):
    query_df = pd.read_excel(query_filename)
    query_dict = dict()
    for col_ in col_list:
        list_ = []
        for elem in query_df[col_].tolist():
            if type(elem) is float and math.isnan(elem):
                #continue
                list_.append("null")
            elif type(elem) is float or type(elem) is int:
                list_.append(str(int(elem)))
            else :
                list_.append(elem)
        query_dict[col_]=list_
    return query_dict

# retrieve row
def retrieve_from_excel(reference_excel_path, query_id, query_list, save_excel_path=None, parser=None):

    excel_pd = pd.read_excel(reference_excel_path)
    key_id = [str(int(elem)) for elem in excel_pd[query_id].tolist()]

    drop_index_list = []
    for ind, query_elem in enumerate(key_id):
        if parser is not None:
            query_elem = parser(query_elem)
        if query_elem in query_list:
            continue
        else:
            drop_index_list.append(ind)
    excel_pd = excel_pd.drop(drop_index_list)
    if save_excel_path is not None:
        excel_pd.to_excel(save_excel_path)
    return excel_pd

def retrieve_n_union_from_excel(target_excel_path, reference_excel_path, save_path, query_id, col_name_list):
    target_df = pd.read_excel(target_excel_path)
    target_id = [str(elem) for elem in target_df[query_id].tolist()]
    reference_df = pd.read_excel(reference_excel_path)
    reference_id = [str(elem) for elem in reference_df[query_id].tolist()]

    for col_name_ in col_name_list:
        tmp_col_list = []
        not_included=[]
        for t_id in target_id:
            if t_id in reference_id:
                t_id_index = reference_id.index(t_id)
                # print("debug", t_id_index, "col_name", col_name_)
                # print("debug2", reference_df[col_name_].tolist()[0])
                col_val = reference_df[col_name_].tolist()[t_id_index]
                tmp_col_list.append(col_val)
                # print("debug", tmp_col_list)
            else:
                not_included.append(t_id)
                tmp_col_list.append("null")
        target_df[col_name_] = tmp_col_list
        # print(col_name_, "not_included", not_included)
    target_df.to_excel(save_path)
    return

"""
target_excel_path = "C:\\Users\\NM\\PycharmProjects\\MDBManager\\query_list.xlsx"
reference_excel_path = "C:\\Users\\NM\\PycharmProjects\\MDBManager\\query_list_sample.xlsx"
save_path = "C:\\Users\\NM\\PycharmProjects\\MDBManager\\query_list_.xlsx"
retrieve_n_union_from_excel(target_excel_path, reference_excel_path, save_path, "ID", ["BAPL score", "Diagnosis code_1"])
"""

# ==============================================================
# tasks related to directory
def make_query_list_from_dir(query_dir, extention="*.nii", parser=None, isdir=False, search_level="one_level"):
    query_list = []
    if len(os.listdir(query_dir)) == 0:
        print("query_dir argument must have any element")
        return
    if search_level == "all_levels":

        for path, subdirs, files in os.walk(query_dir):
            if isdir:
                query_list_ = [os.path.join(path, subdir_) for subdir_ in subdirs]
            else:
                query_list_ = glob.glob(os.path.join(path, extention))
            if parser is None:
                query_list.append(query_list_)
            else :
                query_list.append([str(parser(elem)) for elem in query_list_])
        return np.concatenate(query_list)
    elif search_level == "one_level":
        if isdir:
            query_list = [os.path.join(query_dir, path_) for path_ in os.listdir(query_dir)]
        else:
            query_list = glob.glob(os.path.join(query_dir, extention))
        return np.array(query_list)

def make_pid_excel_from_dir(target_dir, dest_path, filename_parser = None, islabeled=False,
                            count_redundancy=False, pid_col_name="list", extention="*.dcm"):
    """
    :param target_dir: 
    :param dest_path: 
    :param filename_parser: this filename consider extention portion, you have to treat a extention in this parser
    :param islabeled: 
    :param count_redundancy: 
    :param pid_col_name: 
    :param extention: 
    :return: 
    """
    if islabeled is None or count_redundancy is None:
        print("islabeled must not be None")
        return
    if islabeled is True: # labeled nii or separated dcms
        query_list = []
        label = os.listdir(target_dir)
        for cls in label:
            #filelist = os.listdir(os.path.join(target_dir, cls))
            filelist = glob.glob(os.path.join(target_dir, cls, extention))
            if filename_parser:
                pid_list = [filename_parser(os.path.basename(elem)) for elem in filelist]
            else:
                pid_list = [os.path.basename(elem) for elem in filelist]
            #pid_dict[cls] = pid_list
            query_list.append(pid_list)

        query_list = np.concatenate(query_list)
        # count redundancy
        res = Counter(query_list)
        count_dict = dict()
        count_dict[pid_col_name] = [key for key in res]
        if count_redundancy:
            count_dict["count"] = [res[key] for key in res]
        pd_df = pd.DataFrame(count_dict)
        pd_df.to_excel(dest_path)
        return

    elif islabeled is False:
        # not labeled nii or not separated dcm
        #filelist = os.listdir(target_dir)
        filelist = glob.glob(os.path.join(target_dir, extention))
        if filename_parser is not None:
            pid_list = [filename_parser(os.path.basename(elem)) for elem in filelist]
        else:
            pid_list = [os.path.basename(elem) for elem in filelist]

        # count redundancy
        res = Counter(pid_list)
        count_dict = dict()
        count_dict[pid_col_name] = [key for key in res]
        if count_redundancy:
            count_dict["count"] = [res[key] for key in res]
        pd_df = pd.DataFrame(count_dict)
        pd_df.to_excel(dest_path)
    return

def copy_query_list_from_dir(target_dir, dest_dir, query_list=None, query_excel_path=None, query_excel_id = None
                             ,target_query_parser=None, extention="*.nii", isdir=False):
    if query_list is None and query_excel_path is None:
        print("one of variable between query_list, query_excel_path have to be taken")
        return
    if target_query_parser is None:
        target_query_parser = lambda x : x

    target_list = make_query_list_from_dir(query_dir=target_dir, extention=extention, parser=None, isdir=isdir)

    if query_excel_path :
        query_list = make_query_list_from_excel(query_excel_path, query_excel_id, parser=None)
    elif query_list:
        query_list = query_list

    not_included = []
    for target_item in target_list:
        target_item = os.path.basename(target_item)

        if extention[1:] in target_item: # remove extention
            extention_ind = target_item.find(extention[1:])
            target_item = target_item[:extention_ind]

        parsed_target_item = target_query_parser(target_item)
        if parsed_target_item not in query_list:
            not_included.append(target_item)
            continue
        else:
            if isdir :
                target_path = os.path.join(target_dir, target_item)
                dest_path = os.path.join(dest_dir, target_item)
                shutil.copytree(target_path, dest_path)
            else :
                target_path = os.path.join(target_dir, target_item+extention[1:])
                dest_path = os.path.join(dest_dir, target_item+extention[1:])
                shutil.copy(target_path, dest_path)
    print("not_included target_obj from excel list", not_included)
    return

# to reply to the IRM tool
def filename_parser_for_SortDB_Anonymity(filename):
    tags = filename.split("_")
    if len(tags) != 7 :
        print("filename doesn't have intended form of tags")
        return
    else :
        del tags[5]
        return "_".join(tags)


def copy_with_labeling(source_dir, dest_dir, query_info, extention="*.nii", filename_parser=None, excel_parser=None, isdir=False, search="local"):
    """
    :param source_dir: 
    :param dest_dir: 
    :param query_info: data shape is (2, N), (0, N) means identity for each samples, (1, N) means the label for each of samples. 
    :param filename_parser: 
    :return: 
    """
    if excel_parser is None:
        excel_parser = lambda x: x
    dir_query = make_query_list_from_dir(source_dir, extention=extention, isdir=isdir, search_level="one_level")
    print("dir_query", dir_query)
    label = np.unique(query_info[1])
    for label_ in label:
        if os.path.isdir(os.path.join(dest_dir, label_)):
            continue
        else:
            os.mkdir(os.path.join(dest_dir, label_))
    for query_ in dir_query:
        basename = os.path.basename(query_)
        parsed_basename = filename_parser(basename)

        for pid_, label_ in zip(query_info[0], query_info[1]):
            if excel_parser(pid_) not in parsed_basename:
                continue
            else :
                source_path = query_
                target_path = os.path.join(dest_dir, label_, basename)
                if isdir :
                    shutil.copytree(source_path, target_path)
                else:
                    shutil.copy(source_path, target_path)
    return

def sort_dcm_files(target_dir, dest_dir, dest_filename_parser=None, is_rename=False):
    """
    :param target_dir: 
    :param dest_dir: 
    :param dest_filename_parser: this parser need 3 arguments, which means  and used like parser_(source_dcm, source_basename, ind)
    :param is_rename: 
    :return: 
    """
    if dest_filename_parser is None:
        dest_filename_parser = lambda source_basename, ind, dcm_obj=None : source_basename.split(".")[0]+"_"+str(ind)
    query_list = make_query_list_from_dir(target_dir, extention="*.dcm", parser=None, isdir=False)

    #print("debug", query_list)
    dcm_dict = dict()
    # construct patient dict
    for query_ in query_list:
        dcm = pydicom.read_file(query_)
        #print("debug", dcm[0x10, 0x20].value)
        try:
            dcm_dict[dcm[0x10, 0x20].value].append((dcm, query_))
        except KeyError as ke :
            dcm_dict[dcm[0x10, 0x20].value] = [(dcm, query_)]
        #print(dcm.ImagePositionPatient[2])
    # sorting and save
    for key_ in dcm_dict.keys():
        dcm_dict[key_].sort(key=lambda x: int(x[0].ImagePositionPatient[2]))

        for ind, element_ in enumerate(dcm_dict[key_]):
            source_dcm = element_[0]
            source_path = element_[1]
            source_basename = os.path.basename(source_path)
            dest_basename = dest_filename_parser(source_basename, ind, dcm_obj = source_dcm)


            dest_path = os.path.join(dest_dir, dest_basename+".dcm")
            if is_rename :
                os.rename(source_path, dest_path)
            else:
                shutil.copy(source_path, dest_path)
    return

if __name__ == "__main__":
    # def dest_filename_parser(source_dcm, source_basename, ind):
    #     tags = source_basename.split("_")
    #
    #     source_dcm_tags = source_dcm[0x10, 0x20].value.split("_")
    #     tags[-1] = source_dcm_tags[-1]+"_"+str(ind+1)
    #     return "_".join(tags)
    # target_dir = "C:\\Users\\NM\\PycharmProjects\\datas_\\FBB_BRAIN\\FBB_BRAIN_STCT_ANONYMIZED_TOTAL_RAWCN_경성대 반출용\\backup\\FBB_BRAIN_ST_ANONYMIZED_TOTAL_RAW_경성대반출"
    # dest_dir = target_dir
    # p_full_path = [os.path.join(target_dir, subdir_) for subdir_ in os.listdir(target_dir)]
    # for p_path_ in p_full_path:
    #     sort_dcm_files(p_path_ , p_path_ , dest_filename_parser=dest_filename_parser, is_rename=True)

    # print("hello")
    # query_filename = "C:\\Users\\NM\\PycharmProjects\\UNITTEST_\\DB_manager\\query_list.xlsx"
    # query_col = "PatientID"
    # query_list = make_query_list_from_excel(query_filename, [query_col])
    # #query_list = np.concatenate(query_list)
    # print("query_list", query_list)

    # source_excel_dir = "C:\\Users\\NM\\PycharmProjects\\UNITTEST_\\DB_manager"
    # excel_filename = "Amyloid PET dataset_20151201_20180518_SCD MCI AD Normal control case_final_재확인_최종확인.xlsx"
    # excel_p_id_col_name = "ID"
    #
    # excel_df = pd.read_excel(os.path.join(source_excel_dir, excel_filename))
    # refer_list = make_query_list_from_excel(os.path.join(source_excel_dir, excel_filename), [excel_p_id_col_name])
    # print("refer_list", refer_list)
    #
    # dir_query_list = make_query_list_from_dir("C:\\Users\\NM\\PycharmProjects\\UNITTEST_", extention="*.nii")
    # print("dir_query_list", dir_query_list)
    # #print(glob.glob("C:\\Users\\NM\\PycharmProjects\\UNITTEST_", recursive=True))

    # print("retrieve query list from dir")
    # source_excel_dir = "E:\\hkang\\amyloid\\total_data\\static_preprocess\\metadata"
    # excel_filename = "Amyloid PET dataset_20151201_20190215_all patients_final_ 20190227 권유진입력_.xlsx"
    # reference_excel_path = os.path.join(source_excel_dir, excel_filename)
    # excel_pd = pd.read_excel(reference_excel_path)
    # excel_p_id_col_name = "ID"
    # print(excel_pd.index[0])
    #
    # query_dir = "E:\\hkang\\amyloid\\total_data\\static_preprocess\\data\\bapl_nii_180903_results"
    # # dir_query_list = make_query_list_from_dir(query_dir, extention="*.nii")
    # # print("dir_query_list", dir_query_list)
    #
    # target_dir = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\RCTU_t-SNE\\dataset\\bapl_nii_180903_results_after_jyyoung_labeling"
    # dest_path = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\RCTU_t-SNE\\dataset\\bapl_nii_180903_results_after_jyyoung_labeling_list.xlsx"
    # #
    # def parser_(filename):
    #     if "_" in filename:
    #         return filename[2:-8]
    #     elif "x" in filename:
    #         return filename[3:-6]
    # make_pid_excel_from_dir(target_dir, dest_path, filename_parser=parser_, islabeled=True,
    #                         count_redundancy=True, pid_col_name="ID", extention="*.nii")

    # target_excel_path = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\RCTU_t-SNE\\dataset\\Slice grading_transpose.xlsx"
    # #target_excel_path = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\FBB_BRAIN_PET_Anonymize_total_dictionary_newid_190513정리내용_test.xlsx"
    # reference_excel_path = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\Amyloid PET dataset_20151201_20190215_all patients_final_ 20190227 권유진입력_.xlsx"
    # #save_excel_path = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\total_labeled.xlsx"
    # save_excel_path = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\RCTU_t-SNE\\dataset\\Slice grading_transpose_with_label.xlsx"
    #
    # retrieve_n_union_from_excel(target_excel_path, reference_excel_path, save_excel_path, "ID", ["Diagnosis code_1"])


    # def parser_(filename):
    #     if "_" in filename:
    #         return filename[2:-8]
    #     elif "x" in filename:
    #         return filename[3:-6]

    def parser_(filename):
        tag_list = filename.split("_")
        del tag_list[5]
        return "_".join(tag_list)
    query_filename = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\PET\\dataset\\FBB_8\\FBB_BRAIN_PET_Anonymize_total_dictionary_newid_중복제거_label_경성대반출용_Dx&bapl.xlsx"
    dict = make_col_list_from_excel(query_filename, ["PatientID(new)", "BAPL score"])
    print(dict)
    target_dir = "E:\\hkang\\amyloid\\total_data\\static_preprocess\\FBB_BRAIN_ST_ANONYMIZED_TOTAL_CN"
    dest_dir = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\PET\\dataset\\FBB_9\\nii(processed)"
    copy_with_labeling(target_dir, dest_dir, query_info=[dict['PatientID(new)'], dict["BAPL score"]], extention="*.nii",
                       filename_parser=parser_, isdir=False, search="local")

    # query_filename = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\RCTU_t-SNE\\dataset\\rctu_list.xlsx"
    # query_list_dict = make_col_list_from_excel(query_filename, ["bapl1", "bapl2", "bapl3"])
    # query_list = list(query_list_dict["bapl1"]) + list(query_list_dict["bapl2"])+list(query_list_dict["bapl3"])
    #
    # reference_excel_path = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\RCTU_t-SNE\\dataset\\bapl_nii_180903_results_after_jyyoung_labeling_list_with_label.xlsx"
    # query_id = "ID"
    # save_excel_path = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\RCTU_t-SNE\\dataset\\rctu_query_list_with_label.xlsx"
    # retrieve_from_excel(reference_excel_path, query_id, query_list, save_excel_path=save_excel_path, parser=None)