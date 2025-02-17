import pandas as pd
import numpy as np
import os, pickle, time, sys, configparser
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from scipy import stats
import nibabel as nib

from utils_local import reorder_betas_feats_train, reorder_betas_feats_test


exp_config = configparser.ConfigParser()
exp_config.read('../config')
NSD_top_dir = exp_config['DIR']['NSD_top_dir']
comb_df = pd.read_csv(os.path.join(NSD_top_dir,
                      'Scripts',
                      'submit_script',
                      'array_info',
                      'CV_Ridge_df_control_allsubs.csv'))
curr_row = int(sys.argv[1])-1
curr_sub = int(comb_df.iloc[curr_row]['SUB'])
bulk_ind = int(comb_df.iloc[curr_row]['Bulk'])
curr_roi = comb_df.iloc[curr_row]['ROI']
modality = 'img'

size_bulk_vert = int(exp_config['PARAM']['CV_bulk_size'])
start_ind = bulk_ind*size_bulk_vert

results_dir = os.path.join(NSD_top_dir, 'results')
intermediate_top_dir = os.path.join(NSD_top_dir, 'intermediate')

# Need to find out the total number of bulks for each subject
# Load masks
if curr_roi == 'vvs':
    mask_dir = os.path.join(NSD_top_dir,
                            'data',
                            'nsddata',
                            'freesurfer',
                            'subj0{}'.format(curr_sub),
                            'label')
    roi_name = 'streams.mgz'

    def load_vvs_mask(roi, mask_dir):
        _lh_mask_dir = os.path.join(mask_dir, 'lh.{}'.format(roi))
        _rh_mask_dir = os.path.join(mask_dir, 'rh.{}'.format(roi))

        _lh_mask_obj = nib.load(_lh_mask_dir)
        _lh_mask = _lh_mask_obj.get_fdata().squeeze()
        lh_mask = _lh_mask == 5

        _rh_mask_obj = nib.load(_rh_mask_dir)
        _rh_mask = _rh_mask_obj.get_fdata().squeeze()
        rh_mask = _rh_mask == 5
        print('{}, lh {}, rh {}'.format(roi, np.sum(lh_mask), np.sum(rh_mask)))
        return lh_mask, rh_mask


    roi_lh_mask, roi_rh_mask = load_vvs_mask(roi_name, mask_dir)
elif curr_roi == 'A1':
    mask_dir = os.path.join(intermediate_top_dir,
                            'masks',
                            'HCP_MMP_ROIs')
    roi_lh_mask = np.load(os.path.join(mask_dir, 'lh.A1.sub0{}.npy'.format(curr_sub)))
    roi_rh_mask = np.load(os.path.join(mask_dir, 'rh.A1.sub0{}.npy'.format(curr_sub)))

total_vertices = np.sum(roi_lh_mask) + np.sum(roi_rh_mask)
end_ind = np.min([(bulk_ind+1)*size_bulk_vert, total_vertices])

# Set output directory
betas_dir = os.path.join(intermediate_top_dir, 'betas_native')
feat_dir = os.path.join(intermediate_top_dir, 'features', 'CLIP_features')
img_list_dir = os.path.join(intermediate_top_dir, 'features')
weights_dir = os.path.join(results_dir,
                              'CLIP_model_weights',
                              'RidgeWeights_sub0{}_{}_{}to{}_{}_norm'.format(curr_sub, modality,
                                                                    start_ind, end_ind, curr_roi))
intercepts_dir = os.path.join(results_dir,
                        'CLIP_model_weights',
                        'RidgeIntercepts_sub0{}_{}_{}to{}_{}_norm'.format(curr_sub, modality,
                                                                    start_ind, end_ind, curr_roi))
CV_results_dir = os.path.join(results_dir,
                              'CV_results_CLIP',
                              'sub0{}_{}_{}_GS_norm.csv'.format(curr_sub, modality, curr_roi))

# Load features
unique_features_array = np.load(os.path.join(feat_dir,
                    '{}_sub0{}_unique.npy'.format(modality,
                                                  curr_sub)))
share_features_array = np.load(os.path.join(feat_dir,
                    '{}_sub0{}_share.npy'.format(modality,
                                                  curr_sub)))

# Load betas
# Unique images
_bulk_size = int(exp_config['PARAM']['beta_bulk_size'])
unique_imgs_list=np.load(os.path.join(img_list_dir, 'sub0{}_unique_img_order.npy'.format(curr_sub)))
_num_bulks = int(np.ceil(len(unique_imgs_list)/_bulk_size))
list_betas_unique = []
for curr_bulk in range(_num_bulks):
    _start_ind = curr_bulk*_bulk_size
    _end_ind = (curr_bulk+1)*_bulk_size
    curr_bulk_dir = os.path.join(betas_dir,
                                'sub0{}_betas_{}to{}_{}_norm'.format(curr_sub,
                                               _start_ind,
                                                _end_ind,
                                                curr_roi))
    with open(curr_bulk_dir, "rb") as f:
        curr_beta = pickle.load(f)

    if end_ind < total_vertices:
        sampled_vert = [k[start_ind:end_ind, :] for k in curr_beta]
    else:
        sampled_vert = [k[start_ind:, :] for k in curr_beta]
    list_betas_unique = list_betas_unique + sampled_vert

# Shared images
share_imgs_list=np.load(os.path.join(img_list_dir, 'sub0{}_share_img_order.npy'.format(curr_sub)))
_num_bulks = int(np.ceil(len(share_imgs_list)/_bulk_size))
list_betas_share = []
for curr_bulk in range(_num_bulks):
    _start_ind = curr_bulk*_bulk_size
    _end_ind = (curr_bulk+1)*_bulk_size
    curr_bulk_dir = os.path.join(betas_dir,
                                'sub0{}_betas_{}to{}_{}_share_norm'.format(curr_sub,
                                               _start_ind,
                                                _end_ind,
                                                curr_roi))
    with open(curr_bulk_dir, "rb") as f:
        curr_beta = pickle.load(f)
    if end_ind < total_vertices:
        sampled_vert = [k[start_ind:end_ind, :] for k in curr_beta]
    else:
        sampled_vert = [k[start_ind:, :] for k in curr_beta]
    list_betas_share = list_betas_share + sampled_vert

# Create a grid search object
parameters = {'alpha': [1, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6]}
clf = GridSearchCV(Ridge(), parameters, cv=3)
all_weights_array = np.zeros((unique_features_array.shape[1], list_betas_unique[0].shape[0]))
all_intercepts_array = np.zeros((1, list_betas_unique[0].shape[0]))

# Go through all the vertices
for vi in range(list_betas_unique[0].shape[0]):
    time1 = time.time()
    ordered_train_betas, ordered_train_feats = reorder_betas_feats_train(list_betas_unique, unique_features_array,
                                                                         list(range(len(list_betas_unique))))
    # Remove NaNs
    if np.sum(np.isnan(ordered_train_betas[:, vi]))>0:
        print('Detect NaN')
        nan_bool = ~np.isnan(ordered_train_betas[:, vi])
        ordered_train_betas = ordered_train_betas[nan_bool, :]
        ordered_train_feats = ordered_train_feats[nan_bool, :]
        print(ordered_train_betas.shape[0])

    ordered_test_betas, ordered_test_feats = reorder_betas_feats_test(list_betas_share, share_features_array,
                                                                      list(range(len(list_betas_share))))

    # Remove NaNs
    if np.sum(np.isnan(ordered_test_betas[:, vi]))>0:
        print('Detect NaN')
        nan_bool = ~np.isnan(ordered_test_betas[:, vi])
        ordered_test_betas = ordered_test_betas[nan_bool, :]
        ordered_test_feats = ordered_test_feats[nan_bool, :]
        print(ordered_test_betas.shape[0])

    clf.fit(ordered_train_feats,
              ordered_train_betas[:, vi])
    curr_CV_r = stats.pearsonr(clf.predict(ordered_test_feats),
                               ordered_test_betas[:, vi])[0]

    time2 = time.time()

    print(time2 - time1)

    with open(CV_results_dir, 'a') as f:
        f.write('{}, {}, {}, {}\n'.format(curr_sub, start_ind+vi, curr_CV_r, clf.best_params_))
    
     # Save the weights and intercepts
    all_weights_array[:, vi] = clf.best_estimator_.coef_
    all_intercepts_array[:, vi] = clf.best_estimator_.intercept_
    
np.save(weights_dir, all_weights_array)
np.save(intercepts_dir, all_intercepts_array)
        


 


