import torch
import config
from training import train_vqa, gif_preproc, text_preproc
from model import get_PTVQA, get_IQA, get_ImageFeatureExtractor, get_language_encoder
from datasets import get_dataset

#
# Train VQA
#
train_dataset, val_dataset = get_dataset(
   data_source = "IQA",
   dataset_folder = config.coco_folder_location,
   annotation_file = config.vqa_annotation_file_location
   )

# train_dataset, val_dataset, _ = get_dataset(
#    data_source = "TGIF",
#    dataset_folder = config.tgif_folder_location,
#    annotation_file = config.tgif_folder_location + '/action_question_annotations.csv'
#    )

model = get_IQA()

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

best_val_loss=1000
if config.resume_checkpoint:
   save_path = f"checkpoints/model_{config.check_point_load}.pth"
   checkpoint_data = torch.load(save_path)
   best_val_loss = checkpoint_data["best_loss"]
   model.load_state_dict(checkpoint_data['state_dict'])
   print("=> loaded checkpoint '{}' (epoch {}). Best valditation loss: {}"
         .format(config.check_point_load, checkpoint_data['epoch'], best_val_loss))   

if torch.cuda.device_count() > 1 and config.use_gpu:
   print(f"Using multi-gpu: Devices={config.number_devices}")
   config.device = torch.cuda.current_device()
   model.to(config.device)
   model = torch.nn.DataParallel(module=model, device_ids = [i for i in range(torch.cuda.device_count())]).cuda()
else:
   model.to(config.device)

# This needs to be done after moving the model to the gpu 
if config.resume_checkpoint:
   optimizer.load_state_dict(checkpoint_data['optimizer'])
   print("=> loaded optimizer")   

train_vqa(model, optimizer, train_dataset=train_dataset, val_dataset=val_dataset, num_epochs=100, best_val_loss=best_val_loss)

#
# Gif preproc
#
# gif_dataset = get_dataset(
# 	data_source = "GIF_preproc",
# 	dataset_folder = config.tgif_folder_location + "gifs/",
# 	annotation_file = None
# 	)
# model = get_ImageFeatureExtractor()
# gif_preproc(model=model, dataset=gif_dataset, save_folder="F:/Dev/datasets/tgif/tgif_image_features")


#
# Text preproc
#
# text_dataset = get_dataset(
# 	data_source = "text_preproc",
# 	annotation_file = config.tgif_folder_location + "action_question_annotations.csv"
# 	)
# model = get_language_encoder()
# text_preproc(lan_model=model, dataset=text_dataset, save_folder="F:/Dev/datasets/tgif_text_features")