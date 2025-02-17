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
import nibabel as nib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

from utils_local import reorder_betas_feats_train, reorder_betas_feats_test

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
                      'ForwardModel_mean_CLIP_sub_control.csv'))

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
lh_CV_r = np.load(os.path.join(results_dir, 'Anatomical', "lh.sub0{}_wholebrain_r.npy".format(curr_sub)))
rh_CV_r = np.load(os.path.join(results_dir, 'Anatomical', "rh.sub0{}_wholebrain_r.npy".format(curr_sub)))

# Load features
unique_features_array = np.load(os.path.join(feat_dir,
                    '{}_sub0{}_unique.npy'.format(modality,
                                                  curr_sub)))
share_features_array = np.load(os.path.join(feat_dir,
                    '{}_sub0{}_share.npy'.format(modality,
                                                  curr_sub)))


# Load the betas
# Make a mask that excludes vertices that we already looked at
def load_streams_mask(roi, mask_dir, roi_ind):
    _lh_mask_dir = os.path.join(mask_dir, 'lh.{}'.format(roi))
    _rh_mask_dir = os.path.join(mask_dir, 'rh.{}'.format(roi))

    _lh_mask_obj = nib.load(_lh_mask_dir)
    _lh_mask = _lh_mask_obj.get_fdata().squeeze()
    lh_mask = _lh_mask == roi_ind

    _rh_mask_obj = nib.load(_rh_mask_dir)
    _rh_mask = _rh_mask_obj.get_fdata().squeeze()
    rh_mask = _rh_mask == roi_ind

    return lh_mask, rh_mask

mask_dir = os.path.join(NSD_top_dir,
                            'data',
                            'nsddata',
                            'freesurfer',
                            'subj0{}'.format(curr_sub),
                            'label')

intermediate_dir = os.path.join(NSD_top_dir,
                                'intermediate')

roi_name = 'streams.mgz'
roi_lh_mask_vvs, roi_rh_mask_vvs = load_streams_mask(roi_name, mask_dir, 5)

mask_dir = os.path.join(intermediate_dir,
                            'masks',
                            'HCP_MMP_ROIs')
roi_lh_mask_A1 = np.load(os.path.join(mask_dir, 'lh.A1.sub0{}.npy'.format(curr_sub)))
roi_rh_mask_A1 = np.load(os.path.join(mask_dir, 'rh.A1.sub0{}.npy'.format(curr_sub)))

lh_frontal_mask = np.load(os.path.join(mask_dir, 'lh.lPFC.sub0{}.npy'.format(curr_sub)))
rh_frontal_mask = np.load(os.path.join(mask_dir, 'rh.lPFC.sub0{}.npy'.format(curr_sub)))

# Make a mask for the remaining verts
lh_mask_used = roi_lh_mask_vvs + roi_lh_mask_A1 + lh_frontal_mask
lh_mask_remain = ~lh_mask_used.astype(bool)

rh_mask_used = roi_rh_mask_vvs + roi_rh_mask_A1 + rh_frontal_mask
rh_mask_remain = ~rh_mask_used.astype(bool)
    
if curr_roi == 'dvs':
    mask_dir = os.path.join(NSD_top_dir,
                            'data',
                            'nsddata',
                            'freesurfer',
                            'subj0{}'.format(curr_sub),
                            'label')
    _roi_lh_mask, _roi_rh_mask = load_streams_mask(roi_name, mask_dir, 7)
elif curr_roi == 'lvs':
    mask_dir = os.path.join(NSD_top_dir,
                            'data',
                            'nsddata',
                            'freesurfer',
                            'subj0{}'.format(curr_sub),
                            'label')
    _roi_lh_mask, _roi_rh_mask = load_streams_mask(roi_name, mask_dir, 6)
else:
    mask_dir = os.path.join(intermediate_dir,
                                'masks',
                                'HCP_MMP_ROIs')
    _roi_lh_mask = np.load(os.path.join(mask_dir, 'lh.{}.sub0{}.npy'.format(curr_roi, curr_sub)))
    _roi_rh_mask = np.load(os.path.join(mask_dir, 'rh.{}.sub0{}.npy'.format(curr_roi, curr_sub)))

roi_lh_mask = _roi_lh_mask * lh_CV_r > 0.1
roi_rh_mask = _roi_rh_mask * rh_CV_r > 0.1

# Note that the selected verts need to be indexing the remaining verts
lh_remain_verts_num= np.sum(lh_mask_remain)
final_selected_verts_lh = np.where(roi_lh_mask[lh_mask_remain])[0]
final_selected_verts_rh = np.where(roi_rh_mask[rh_mask_remain])[0]+lh_remain_verts_num
final_selected_verts = np.hstack((final_selected_verts_lh, final_selected_verts_rh))
print('{}, {}'.format(curr_roi, len(final_selected_verts)))

# Unique images
_bulk_size = 100
unique_imgs_list=np.load(os.path.join(img_list_dir, 'sub0{}_unique_img_order.npy'.format(curr_sub)))
_num_bulks = int(np.ceil(len(unique_imgs_list)/_bulk_size))
list_betas_unique = []
for curr_bulk in range(_num_bulks):
    _start_ind = curr_bulk*_bulk_size
    _end_ind = (curr_bulk+1)*_bulk_size
    
    curr_bulk_dir = os.path.join(betas_dir,
                            'sub0{}_betas_{}to{}_all_norm'.format(curr_sub,
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

    curr_bulk_dir = os.path.join(betas_dir,
     'sub0{}_betas_{}to{}_all_share_norm'.format(curr_sub,
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