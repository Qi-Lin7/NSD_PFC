import pandas as pd
import numpy as np
import scipy.io as sio
import os, time, h5py, sys, configparser, pickle
from tqdm import tqdm
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from PIL import Image
import torch
import clip
from utils_local import load_sub_resp_df

exp_config = configparser.ConfigParser()
exp_config.read('../config')
NSD_top_dir = exp_config['DIR']['NSD_top_dir']
comb_df = pd.read_csv(os.path.join(NSD_top_dir,
                      'Scripts',
                      'submit_script',
                      'array_info',
                      'Feature_Extraction_df_CLIP.csv'))
curr_row = int(sys.argv[1])-1
curr_sub = int(comb_df.iloc[curr_row]['SUB'])
layer_name = 'image'

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(os.path.join(NSD_top_dir, 'models', 'ViT-B-32.pt'))
model.eval()

# Set the directories
stimuli_dir = os.path.join(NSD_top_dir,
                           'data',
                          'nsddata_stimuli',
                          'stimuli',
                          'nsd',
                          'nsd_stimuli.hdf5')
intermediate_dir = os.path.join(NSD_top_dir, 'intermediate', 'features')
output_dir = os.path.join(intermediate_dir, 'CLIP_features')

# Load stimuli info

# Load example image
f1 = h5py.File(stimuli_dir,'r')
tmp_img_array = f1['imgBrick'][10, :, :, :]
tmp_img_obj = Image.fromarray(np.uint8(tmp_img_array)).convert('RGB')
tmp_image_input = preprocess(tmp_img_obj).unsqueeze(0).to(device)
with torch.no_grad():
    tmp_image_features = model.encode_image(tmp_image_input)
print(tmp_image_features.shape)

# Load share images
share_imgs_list = np.load(os.path.join(intermediate_dir, 'sub0{}_share_img_order.npy'.format(curr_sub)))
all_img_act = np.zeros((len(share_imgs_list), tmp_image_features.shape[1]), dtype=np.float16)
for ind, curr_img in tqdm(enumerate(share_imgs_list)):
    tmp_img_array = f1['imgBrick'][curr_img-1, :, :, :]
    tmp_img_obj = Image.fromarray(np.uint8(tmp_img_array)).convert('RGB')
    tmp_image_input = preprocess(tmp_img_obj).unsqueeze(0).to(device)
    with torch.no_grad():
        tmp_image_features = model.encode_image(tmp_image_input)
    all_img_act[ind, :] = tmp_image_features
all_img_act = all_img_act.astype(np.float16)
npy_output_dir = os.path.join(output_dir, 'img_sub0{}_share.npy'.format(curr_sub))
np.save(npy_output_dir, all_img_act)

# Load unique images
unique_imgs_list=np.load(os.path.join(intermediate_dir, 'sub0{}_unique_img_order.npy'.format(curr_sub)))
all_img_act = np.zeros((len(unique_imgs_list),  tmp_image_features.shape[1]), dtype=np.float16)
for ind, curr_img in tqdm(enumerate(unique_imgs_list)):
    tmp_img_array = f1['imgBrick'][curr_img-1, :, :, :]
    tmp_img_obj = Image.fromarray(np.uint8(tmp_img_array)).convert('RGB')
    tmp_image_input = preprocess(tmp_img_obj).unsqueeze(0).to(device)
    with torch.no_grad():
        tmp_image_features = model.encode_image(tmp_image_input)
    all_img_act[ind, :] = tmp_image_features
all_img_act = all_img_act.astype(np.float16)
npy_output_dir = os.path.join(output_dir, 'img_sub0{}_unique.npy'.format(curr_sub))
np.save(npy_output_dir, all_img_act)
f1.close()

