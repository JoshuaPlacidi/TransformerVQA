# file to store configuration details of model

device = 'cuda:0'#"cpu"

batch_size = 3

image_size = (256,256)

h_dim = 256
padded_frame_length = 20
padded_language_length = 10



# dataset locations
#
#   folders should have the following structure:
#   dataset_name
#    |--> videos (folder containing ALL videos)
#    |--> train_annotations.csv (csv file containing all training sample information)
#    |--> test_annotations.csv (csv file containing all testing sample information)
#
tgif_folder_location = "F:/dev/datasets/tgif/"
