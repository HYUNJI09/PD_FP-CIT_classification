import os
import shutil
import glob
import platform
import pandas as pd
import openpyxl


################################## USER'S CONTROL [START] ##################################
# 1. 해당 경로 내에 있는 파일 목록을 불러온다.
test_path = r'E:\0_1_FBB-FDG_추가_2021\0_Raw_Preprocessed\1_FBB_early\0_3_FBB_DY_0-20_cnt_SUVr_74\그외'

excel_list_path = r'E:\0_1_FBB-FDG_추가_2021\MCI_2021.xlsx'

copy_path = r'E:\0_1_FBB-FDG_추가_2021\0_Raw_Preprocessed\1_FBB_early\0_3_FBB_DY_0-20_cnt_SUVr_74\MCI'

# 1-1. 특정 확장자만 불러올 것
ext_cond = "*.xlsx" #"*.xlsx" "*.nii"
################################## USER'S CONTROL [ END ] ##################################

# 1. 리스트 만들기
## 1-1. 경로 내에 있는 파일 리스트 만들기
full_data_path = glob.glob(os.path.join(test_path, ext_cond))     # os.listdir(test_path)

data_list = []
for data_ in full_data_path:
    filename = os.path.basename(data_)
    data_list.append(filename.split("_")[0])
data_list.sort()

print(data_list)

## 1-2. 엑셀 내 리스트 만들기
df = pd.read_excel(excel_list_path, engine='openpyxl') # 엑셀 위치
df_ = df['Patient ID'].values
df_list= df_.tolist()

excel_list = []
for i in df_list:
    excel_list.append(str(i))

print(excel_list)

# 2. 파일 리스트와 엑셀 리스트가 동일하면 복사하기
for ind, i in enumerate(data_list):
    for f in excel_list:
        if f == i:
            filename = os.path.basename(full_data_path[ind])
            shutil.move(os.path.join(test_path, filename), os.path.join(copy_path, filename))

print("[!] Process finished.")
