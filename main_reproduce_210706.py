import sys
import numpy as np

from DataIO import NIIDataIO

if __name__ == "__main__":
    print("test")

    task_mode = "3D_CNN_model_test_with_multiple_comparison"  # "3D_CNN_model_test_with_multiple_comparison" "3D_CNN_model_test_with_ft_sel"  # "3D_CNN_model_sel" / "3D_CNN_model_test" / "SVM" / "DNN_experiment" / "3D_CNN_model_test_with_ft_sel"

    # Data loader
    # Loading data
    data_dir = r"C:\Users\shyun\PycharmProjects\datas\1_PD_test"
    extension = "nii"
    dataDim = "3D"  # 3D
    view = "axial"

    channel_size = 1

    nii_io = NIIDataIO()
    nii_io.load_v2(data_dir, extension=extension, data_dir_child="labeled",
                   dataDim=dataDim, instanceIsOneFile=True, channel_size=channel_size, view=view)

    datas = nii_io._3D_data  # (N, D, W, H), (397, 110, 200, 200)
    labels = nii_io._label  # (N, ) [0, 1, ...], (397, )
    data_filenames = nii_io._3D_data_filename  # (N, ), (397, )
    label_name = nii_io._label_name  # (N, ) ['0_NC', '1_PD', ..], (397, )
    class_name = nii_io._class_name  # (Num_classes, ), (2, ), ['0_NC' '1_PD']
    print("_3D_data", np.array(nii_io._3D_data).shape)  # (213, 68, 95, 79, 1)
    print("_3D_data_filename", np.array(nii_io._3D_data_filename).shape)  # (256,)
    print(np.array(nii_io._label).shape)  # (256,)
    print(np.array(nii_io._label_name).shape)  # (256,)
    print(np.array(nii_io._class_name))  # ['0_bapl1' '1_bapl23']