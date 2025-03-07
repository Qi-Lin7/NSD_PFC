import pandas as pd
import numpy as np
import os, pickle, time, sys, random, configparser
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from scipy import stats
from tqdm import tqdm

from utils_local import reorder_betas_feats_test

exp_config = configparser.ConfigParser()
exp_config.read('../config')
NSD_top_dir = exp_config['DIR']['NSD_top_dir']
comb_df = pd.read_csv(os.path.join(NSD_top_dir,
                      'Scripts',
                      'Manuscript',
                      'submit_script',
                      'array_info',
                      'Bootstrap_CLIP_img_full.csv'))
curr_row = int(sys.argv[1])-1
curr_sub = int(comb_df.iloc[curr_row]['SUB'])
modality = 'img'

size_bulk_vert = int(exp_config['PARAM']['CV_bulk_size'])
results_dir = os.path.join(NSD_top_dir, 'results')
intermediate_top_dir = os.path.join(NSD_top_dir, 'intermediate')
betas_dir = os.path.join(intermediate_top_dir, 'betas_native')
feat_dir = os.path.join(intermediate_top_dir, 'features', 'CLIP_features')
img_list_dir = os.path.join(intermediate_top_dir, 'features')
mask_output_dir = os.path.join(intermediate_top_dir, 'masks', 'HCP_MMP_ROIs')

# Find out the number of bulks
lh_frontal_mask = np.load(os.path.join(mask_output_dir, 'lh.lPFC.sub0{}.npy'.format(curr_sub)))
rh_frontal_mask = np.load(os.path.join(mask_output_dir, 'rh.lPFC.sub0{}.npy'.format(curr_sub)))
total_vertices = np.sum(lh_frontal_mask) + np.sum(rh_frontal_mask)

# Find out the selected verts
selected_vertices_df = pd.read_csv(os.path.join(results_dir,
                            'CLIP_selected_verts',
                            '{}_lPFC_selected_verts.csv'.format(modality)))
curr_sub_df = selected_vertices_df[selected_vertices_df['SUB']==curr_sub]
selected_vertices = curr_sub_df['Vertex'].values
total_bulks = int(np.ceil(len(selected_vertices)/size_bulk_vert))


# Load all the weights
for bulk_ind in range(total_bulks):
    start_ind = bulk_ind*size_bulk_vert
    end_ind = np.min([(bulk_ind+1)*size_bulk_vert, len(selected_vertices)])
    _weights_dir = os.path.join(results_dir,
                              'CLIP_model_weights',
                              'RidgeWeights_sub0{}_{}_{}to{}_norm.npy'.format(curr_sub, modality,
                                                                    start_ind, end_ind))
    curr_weights = np.load(_weights_dir)
    
    _intercepts_dir = os.path.join(results_dir,
                        'CLIP_model_weights',
                        'RidgeIntercepts_sub0{}_{}_{}to{}_norm.npy'.format(curr_sub, modality,
                                                                    start_ind, end_ind))
    curr_intercepts = np.load(_intercepts_dir)
    
    if bulk_ind == 0:
        all_weights_array = curr_weights
        all_intercepts_array = curr_intercepts
    else:
        all_weights_array = np.hstack((all_weights_array, curr_weights))
        all_intercepts_array = np.hstack((all_intercepts_array, curr_intercepts))

# Load features
share_features_array = np.load(os.path.join(feat_dir,
                    '{}_sub0{}_share.npy'.format(modality,
                                                  curr_sub)))
print(share_features_array.shape)

# Load betas

# Shared images
_bulk_size = int(exp_config['PARAM']['beta_bulk_size'])
share_imgs_list=np.load(os.path.join(img_list_dir, 'sub0{}_share_img_order.npy'.format(curr_sub)))
_num_bulks = int(np.ceil(len(share_imgs_list)/_bulk_size))
list_betas_share = []
for curr_bulk in range(_num_bulks):
    _start_ind = curr_bulk*_bulk_size
    _end_ind = (curr_bulk+1)*_bulk_size
    curr_bulk_dir = os.path.join(betas_dir,
                                'sub0{}_betas_{}to{}_share_norm'.format(curr_sub,
                                               _start_ind,
                                                _end_ind))
    with open(curr_bulk_dir, "rb") as f:
        curr_beta = pickle.load(f)
    sampled_vert = [k[selected_vertices, :] for k in curr_beta]
    list_betas_share = list_betas_share + sampled_vert

# Order the betas
ordered_test_betas, ordered_test_feats = reorder_betas_feats_test(list_betas_share, share_features_array,                                                  list(range(len(list_betas_share))))

# Predict the betas from features
predicted_betas = np.matmul(ordered_test_feats, all_weights_array) + all_intercepts_array

# Run bootstrap
random.seed(1)
n_iters = 10000
bootstrap_rs = np.zeros((predicted_betas.shape[1], n_iters))
for curr_iter in tqdm(range(n_iters)):
    choosen_inds = np.array(random.choices(range(predicted_betas.shape[0]), k = predicted_betas.shape[0]))
    sampled_predicted_betas = predicted_betas.copy()[choosen_inds, :]
    #print(sampled_predicted_betas.shape)
    sampled_test_betas = ordered_test_betas.copy()[choosen_inds, :]
    #print(sampled_test_betas.shape)
    _test_df = pd.DataFrame(sampled_test_betas)
    _pred_df = pd.DataFrame(sampled_predicted_betas)
    
    curr_corr = _pred_df.corrwith(_test_df)
    bootstrap_rs[:, curr_iter] = curr_corr
    
# Save the bootstrapped results 

bootstrap_output_dir = os.path.join(results_dir, 'Bootstrap_results')
np.save(os.path.join(bootstrap_output_dir,
                    'sub0{}_{}_{}iter_norm.npy'.format(curr_sub, modality, n_iters)),
       bootstrap_rs)