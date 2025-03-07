# Fit a regression model based on the average activities
import pandas as pd
import numpy as np
import scipy.io as sio
from scipy import stats
import os, time, h5py, sys, configparser, pickle
from tqdm import tqdm
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from PIL import Image
import torch
import clip
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

from utils_local import reorder_betas_feats_test

exp_config = configparser.ConfigParser()
exp_config.read('../config')
NSD_top_dir = exp_config['DIR']['NSD_top_dir']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(os.path.join(NSD_top_dir, 'models', 'ViT-B-32.pt'))
model.eval()

curr_row = int(sys.argv[1])-1
comb_df = pd.read_csv(os.path.join(NSD_top_dir,
                      'Scripts',
                      'submit_script',
                      'array_info',
                      'ForwardModel_mean_CLIP_sub.csv'))

curr_sub = int(comb_df.iloc[curr_row]['SUB'])
curr_roi = comb_df.iloc[curr_row]['ROI']
modality = 'img'

# Set the directories
stimuli_dir = os.path.join(NSD_top_dir,
                           'data',
                          'nsddata_stimuli',
                          'stimuli',
                          'nsd',
                          'nsd_stimuli.hdf5')
img_list_dir = os.path.join(NSD_top_dir,
                           'intermediate',
                           'features')
feat_dir = os.path.join(img_list_dir, 'CLIP_features')
intermediate_top_dir = os.path.join(NSD_top_dir, 
                                    'intermediate')
betas_dir = os.path.join(intermediate_top_dir, 'betas_native')
results_dir = os.path.join(NSD_top_dir, 'results')
summary_df_dir = os.path.join(results_dir,
                            'Pred_activation',
                             'sub0{}_{}.csv'.format(curr_sub,
                                                   curr_roi))

# Load the prediction results to select the vertices
if curr_roi == 'lPFC':
    CV_summary_df = pd.read_csv(os.path.join(results_dir , 
                                'CV_results_CLIP',
                                'sub0{}_img_GS_norm.csv'.format(curr_sub)),
                header=None, names=['SUB', 'VERTEX', 'CV-r', 'best alpha'])
elif curr_roi == 'Visual':
    CV_summary_df = pd.read_csv(os.path.join(results_dir , 
                                'CV_results_CLIP',
                                'sub0{}_img_vvs_GS_norm.csv'.format(curr_sub)),
                header=None, names=['SUB', 'VERTEX', 'CV-r', 'best alpha'])

# Load features
unique_features_array = np.load(os.path.join(feat_dir,
                    '{}_sub0{}_unique.npy'.format(modality,
                                                  curr_sub)))
share_features_array = np.load(os.path.join(feat_dir,
                    '{}_sub0{}_share.npy'.format(modality,
                                                  curr_sub)))


# Load the betas
final_selected_verts = list(CV_summary_df[CV_summary_df['CV-r']>0.1]['VERTEX'].values)

# Unique images
_bulk_size = 100
unique_imgs_list=np.load(os.path.join(img_list_dir, 'sub0{}_unique_img_order.npy'.format(curr_sub)))
_num_bulks = int(np.ceil(len(unique_imgs_list)/_bulk_size))
list_betas_unique = []
for curr_bulk in range(_num_bulks):
    _start_ind = curr_bulk*_bulk_size
    _end_ind = (curr_bulk+1)*_bulk_size
    if curr_roi == 'lPFC':
        curr_bulk_dir = os.path.join(betas_dir,
                                'sub0{}_betas_{}to{}_norm'.format(curr_sub,
                                               _start_ind,
                                                _end_ind))
    elif curr_roi == 'Visual':
        curr_bulk_dir = os.path.join(betas_dir,
                                'sub0{}_betas_{}to{}_vvs_norm'.format(curr_sub,
                                               _start_ind,
                                                _end_ind))
    with open(curr_bulk_dir, "rb") as f:
        curr_beta = pickle.load(f)
    sampled_vert = [np.expand_dims(np.mean(k[final_selected_verts, :], axis=0), axis=0) for k in curr_beta]
    list_betas_unique = list_betas_unique + sampled_vert

# Shared images
share_imgs_list=np.load(os.path.join(img_list_dir, 'sub0{}_share_img_order.npy'.format(curr_sub)))
_num_bulks = int(np.ceil(len(share_imgs_list)/_bulk_size))
list_betas_share = []
for curr_bulk in range(_num_bulks):
    _start_ind = curr_bulk*_bulk_size
    _end_ind = (curr_bulk+1)*_bulk_size
    if curr_roi == 'lPFC':
        curr_bulk_dir = os.path.join(betas_dir,
                                'sub0{}_betas_{}to{}_share_norm'.format(curr_sub,
                                               _start_ind,
                                                _end_ind))
    elif curr_roi == 'Visual':
        curr_bulk_dir = os.path.join(betas_dir,
                                'sub0{}_betas_{}to{}_vvs_share_norm'.format(curr_sub,
                                               _start_ind,
                                                _end_ind))
    with open(curr_bulk_dir, "rb") as f:
        curr_beta = pickle.load(f)
    sampled_vert = [np.expand_dims(np.mean(k[final_selected_verts, :], axis=0), axis=0) for k in curr_beta]
    list_betas_share = list_betas_share + sampled_vert
    
    
# Create a grid search object
parameters = {'alpha': [1, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6]}
clf = GridSearchCV(Ridge(), parameters, cv=3)

ordered_train_betas, ordered_train_feats = reorder_betas_feats_test(list_betas_unique, unique_features_array,
                                                                     list(range(len(list_betas_unique))))
# Remove NaNs
if np.sum(np.isnan(ordered_train_betas))>0:
    print('Detect NaN')
    nan_bool = np.isnan(ordered_train_betas)
    rows_wo_nan = ~(np.sum(nan_bool, axis=1)>0)
    
    ordered_train_betas = ordered_train_betas[rows_wo_nan, :]
    ordered_train_feats = ordered_train_feats[rows_wo_nan, :]
    print(ordered_train_betas.shape[0])

ordered_test_betas, ordered_test_feats = reorder_betas_feats_test(list_betas_share, share_features_array,                                               list(range(len(list_betas_share))))

# Remove NaNs
if np.sum(np.isnan(ordered_test_betas))>0:
    print('Detect NaN')
    nan_bool = ~np.isnan(ordered_test_betas)
    ordered_test_betas = ordered_test_betas[nan_bool]
    ordered_test_feats = ordered_test_feats[nan_bool]
    print(ordered_test_betas.shape[0])

clf.fit(np.vstack((ordered_train_feats, ordered_test_feats)),
          np.vstack((ordered_train_betas, ordered_test_betas)))
print('Finished training!')

clf_output_dir = os.path.join(results_dir, 'Univariate_weighting_avg', '{}_{}_avg_clf_allimgs.pkl'.format(curr_sub, curr_roi))

with open(clf_output_dir, 'wb') as f:
    pickle.dump(clf, f)
    
    
# Apply the classifier to all the images
sub_list = range(1, 9)
# First load all the unique images
for curr_sub in sub_list:
    unique_imgs_list=np.load(os.path.join(img_list_dir, 'sub0{}_unique_img_order.npy'.format(curr_sub)))
    unique_features_array = np.load(os.path.join(feat_dir,
                    '{}_sub0{}_unique.npy'.format(modality,
                                                  curr_sub)))
    if curr_sub == 1:
        all_imgs = unique_imgs_list
        all_features_array = unique_features_array
    else:
        all_imgs = np.hstack((all_imgs, unique_imgs_list))
        all_features_array = np.vstack((all_features_array, unique_features_array))


# Then load the share images (only need to load once)
share_img_list = np.load(os.path.join(img_list_dir, 'sub0{}_share_img_order.npy'.format(1)))
all_imgs = np.hstack((all_imgs, share_img_list))
share_features_array = np.load(os.path.join(feat_dir,
                    '{}_sub0{}_share.npy'.format(modality,
                                                  1)))
all_features_array = np.vstack((all_features_array, share_features_array))
print('Finished loading pre-extracted features!')

# Find out what images are missing
full_img_list = np.array(range(1, 73001))
missing_imgs = full_img_list[~np.isin(full_img_list, all_imgs)]
# Load the missing images
f1 = h5py.File(stimuli_dir,'r')
img_array = f1['imgBrick'][missing_imgs-1, :, :, :]
img_list = list(img_array)
img_obj = [preprocess(Image.fromarray(np.uint8(image)).convert('RGB')) for image in img_list]
img_input = torch.stack(img_obj).to(device)
with torch.no_grad():
    missing_features = model.encode_image(img_input)
    
# Apply the classifer to the features
shown_imgs_act = clf.predict(all_features_array)

df1 = pd.DataFrame(np.squeeze(shown_imgs_act), columns=['activation'])
df1['img'] = all_imgs

missing_imgs_act = clf.predict(missing_features)
df2 = pd.DataFrame(np.squeeze(missing_imgs_act), columns=['activation'])
df2['img'] = missing_imgs

df = df1.append(df2).reset_index(drop=True)
# Save the dataframe
df.to_csv(summary_df_dir)