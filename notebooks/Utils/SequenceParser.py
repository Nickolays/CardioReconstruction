import os


def remove_sequences(main_path, new_path):

    # Итерируемся по всем пациентам
    for address, dirs, files in os.walk(main_path):
        # Все названия внутри каждого пациента
        for name in files:
            if 'sequence' in name:
                cur_dir = address.split('\\')[-1]
                os.rename(
                    os.path.join(address, name),
                    os.path.join(new_path, cur_dir, name)
                )

#
main_path = r'J:\Data'
new_path = r'J:\Data\sequences'
#
train_path = os.path.join(main_path, 'training')
test_path = os.path.join(main_path, 'testing')
#
new_train_path = os.path.join(new_path, 'training')
new_test_path = os.path.join(new_path, 'testing')

# Get all folder's names of patients
# train_folders = os.listdir(train_path)
# test_folders = os.listdir(test_path)

# # Create dirs in new path for test
# for folder_name in train_folders:
#     os.mkdir(os.path.join(new_train_path, folder_name))
# # Create dirs in new path for test
# for folder_name in test_folders:
#     os.mkdir(os.path.join(new_test_path, folder_name))

# remove_sequences(train_path, new_train_path)
# remove_sequences(test_path, new_test_path)
