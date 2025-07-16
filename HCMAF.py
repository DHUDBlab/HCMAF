from train_test import *

if __name__ == "__main__":
    data_folder = 'BRCA'
    print(data_folder)
    view_list = [1,2,3]
    num_epoch = 500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3

    if data_folder == 'ROSMAP':
        num_class = 2
    if data_folder == 'BRCA':
        num_class = 5
    if data_folder == 'LGG':
        num_class = 2
    if data_folder == 'KIPAN':
        num_class = 3
    train_test1(data_folder, view_list, num_class,
                num_epoch)