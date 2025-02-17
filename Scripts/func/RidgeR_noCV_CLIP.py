import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import os, sys, time, pickle, configparser

exp_config = configparser.ConfigParser()
exp_config.read('../config')
NSD_top_dir = exp_config['DIR']['NSD_top_dir']
comb_df = pd.read_csv(os.path.join(NSD_top_dir,
                      'Scripts',
                      'submit_script',
                      'array_info',
                      'NoCV_Ridge_df_lPFC_allsubs.csv'))
curr_row = int(sys.argv[1])-1
curr_sub = int(comb_df.iloc[curr_row]['SUB'])
bulk_ind = int(comb_df.iloc[curr_row]['Bulk'])
modality = 'img'

bulk_size = int(exp_config['PARAM']['NoCV_bulk_size'])
start_vert = bulk_size*bulk_ind
# Need to find out the total number of bulks for each subject
# Load masks
mask_output_dir = os.path.join(NSD_top_dir, 'intermediate', 'masks', 'HCP_MMP_ROIs')
lh_frontal_mask = np.load(os.path.join(mask_output_dir, 'lh.lPFC.sub0{}.npy'.format(curr_sub)))
rh_frontal_mask = np.load(os.path.join(mask_output_dir, 'rh.lPFC.sub0{}.npy'.format(curr_sub)))
total_vertices = np.sum(lh_frontal_mask) + np.sum(rh_frontal_mask)

end_vert = np.min([bulk_size*(bulk_ind+1), total_vertices]) # max number of vertices
intermediate_top_dir =  os.path.join(NSD_top_dir, 'intermediate')
results_df_dir = os.path.join(NSD_top_dir, 'results', 'RSS_CLIP', 'sub0{}_{}_RSS_norm.csv'.format(curr_sub, modality ))
betas_dir = os.path.join(intermediate_top_dir, 'betas_native')
img_list_dir = os.path.join(intermediate_top_dir, 'features')
feat_dir = os.path.join(intermediate_top_dir, 'features', 'CLIP_features')

# Load features
features_array = np.load(os.path.join(feat_dir, 
                    '{}_sub0{}_unique.npy'.format(modality,
                                                  curr_sub)))
# Load betas
beta_bulk_size = int(exp_config['PARAM']['beta_bulk_size'])
unique_imgs_list = np.load(os.path.join(img_list_dir, 'sub0{}_unique_img_order.npy'.format(curr_sub)))
beta_num_bulks = int(np.ceil(len(unique_imgs_list)/beta_bulk_size))
list_betas = []
for curr_bulk in range(beta_num_bulks):
    start_ind = curr_bulk*beta_bulk_size
    end_ind = (curr_bulk+1)*beta_bulk_size
    curr_bulk_dir = os.path.join(betas_dir,
                                'sub0{}_betas_{}to{}_norm'.format(curr_sub,
                                               start_ind,
                                                end_ind))
    with open(curr_bulk_dir, "rb") as f:
        curr_beta = pickle.load(f)
    if end_vert < total_vertices:
        sampled_vert = [k[start_vert:end_vert, :] for k in curr_beta]
    else:
        sampled_vert = [k[start_vert:, :] for k in curr_beta]
    list_betas = list_betas + sampled_vert

# Pad the features
num_patterns = [k.shape[1] for k in list_betas]
all_betas_array = np.zeros((np.sum(num_patterns),list_betas[0].shape[0]))
all_features_array = np.zeros((np.sum(num_patterns), features_array.shape[1]))
count = 0
for ind, curr_betas in enumerate(list_betas):
    curr_num_pat = num_patterns[ind]
    all_betas_array[count:count+curr_num_pat,
                   :] = curr_betas.T
    curr_feature = np.expand_dims(features_array[ind, :], axis=1)
    curr_features_rep = np.repeat(curr_feature, curr_num_pat, axis=1).T
    all_features_array[count:count+curr_num_pat,
                   :] = curr_features_rep
    count += curr_num_pat 

# Run the ridge regression models
model = Ridge()

for curr_vert in range(list_betas[0].shape[0]):
    time1 = time.time()
    curr_beta = all_betas_array[:, curr_vert].copy()
    curr_features_array = all_features_array.copy()
    # First check if there's any nan
    if np.sum(np.isnan(curr_beta))>0:
        print('Detect NaN')
        nan_bool = ~np.isnan(curr_beta)
        curr_beta = curr_beta[nan_bool]
        curr_features_array = curr_features_array[nan_bool, :]
        
    model.fit(curr_features_array, curr_beta)
    curr_rss = model.score(curr_features_array, curr_beta)
    with open(results_df_dir, 'a') as f:
        f.write('{}, {}, {}\n'.format(start_vert+curr_vert,
                                   curr_rss, 
                                    np.sum(np.isnan(curr_beta))))
    time2 = time.time()
    print(time2-time1)