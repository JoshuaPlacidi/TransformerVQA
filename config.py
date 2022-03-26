# file to store configuration details of model
import torch

train_bert = True
train_deit = False
apply_masking_attn = False
percentage_validation = 0.25 # Use only the 1/4 of the validation set
resume_checkpoint = False
learning_rate = 0.00005
feature_extractor = "deit"
encoder_num_layers = 6
batch_size = 50

check_point_name = f"train_bert_{str(train_bert)}_train_deit_{train_deit}_attn_mask_{str(apply_masking_attn)}_lr_{learning_rate}_fe_{feature_extractor}_batch_{batch_size}"

# Specify a different checkpoint to load pre-trained model. Only if resume_checkpoint == True
check_point_load = "train_bert_True_attn_mask_False_lr_5e-05_2"


print(f"\n\nCURRENT EXECUTION: {check_point_name}\n\n. From checkpoint: {resume_checkpoint}")

use_gpu = True
print("Initialization in cofig.py:")
print(f"Cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = torch.cuda.current_device()
# device = "cpu"

number_devices = torch.cuda.device_count()
print(f"Cuda current_device: {device}")
print(f"Cuda device_count: {number_devices}")
batch_size = batch_size * number_devices

image_size = (224,224)

h_dim = 512
padded_frame_length = 20
padded_language_length_question = 14
padded_language_length_answer = 6

# dataset locations
#
#   folders should have the following structure:
#   dataset_name
#    |--> videos (folder containing ALL videos)
#    |--> train_annotations.csv (csv file containing all training sample information)
#    |--> test_annotations.csv (csv file containing all testing sample information)
#
tgif_folder_location = "F:/dev/datasets/tgif/"
tgif_folder_location = "/home/mena/Documents/Master/mlp_sem2/PTVQA/data/tgif/"
iqa_folder_location = "./data/iqa/"
coco_folder_location = "./data/iqa/"
vqa_annotation_file_location = "./data/iqa/annotations_coco.csv"
