import pandas as pd
import os

annos = pd.read_csv('F:/Dev/datasets/tgif/all_action_question_annotations.csv')

img_files = [x.replace('.pkl','') for x in os.listdir('F:/Dev/datasets/tgif/tgif_image_features')]

annos = annos[annos['gif_name'].isin(img_files)]

annos.to_csv('action_question_annotations.csv')