import pandas as pd
import numpy as np
import scipy.io as sio
from scipy import stats
import os, time, h5py, sys, glob, pickle, random, configparser
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from PIL import Image
import torch
import clip
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

import nibabel as nib
from itertools import combinations, product
from utils_local import reorder_betas_feats_test

exp_config = configparser.ConfigParser()
exp_config.read('../config')
NSD_top_dir = exp_config['DIR']['NSD_top_dir']

curr_seed = int(sys.argv[1])
random.seed(curr_seed)
# Set the directories
results_top_dir = os.path.join(NSD_top_dir, 'results')
img_list_dir = os.path.join(NSD_top_dir, 
                            'intermediate', 
                            'features')
feat_dir = os.path.join(img_list_dir, 'CLIP_features')
intermediate_top_dir = os.path.join(NSD_top_dir, 
                                    'intermediate')
betas_dir = os.path.join(intermediate_top_dir, 'betas_native')
stimuli_dir = os.path.join(NSD_top_dir,
                           'data',
                          'nsddata_stimuli',
                          'stimuli',
                          'nsd',
                          'nsd_stimuli.hdf5')
results_dir = os.path.join(NSD_top_dir, 'results')
summary_df_dir_rsa = os.path.join(results_dir,
                            'Subsampling_results',
                             'summary_rsa_range_{}.csv'.format(curr_seed))
summary_df_dir_univariate = os.path.join(results_dir,
                            'Subsampling_results',
                             'summary_uni_range_{}.csv'.format(curr_seed))

sub_list = range(1, 9)
roi_list = ['vvs', 'lPFC']

# Load the prediction results to select the vertices
count = 0
for curr_sub in sub_list:
    for curr_roi in roi_list:
        if curr_roi == 'lPFC':
            tmp_CV_summary_df = pd.read_csv(os.path.join(results_dir , 
                                'CV_results_CLIP',
                                'sub0{}_img_GS_norm.csv'.format(curr_sub)),
                header=None, names=['SUB', 'VERTEX', 'CV-r', 'best alpha'])
        elif curr_roi == 'vvs':
            tmp_CV_summary_df = pd.read_csv(os.path.join(results_dir , 
                                'CV_results_CLIP',
                                'sub0{}_img_vvs_GS_norm.csv'.format(curr_sub)),
                header=None, names=['SUB', 'VERTEX', 'CV-r', 'best alpha'])
        tmp_CV_summary_df['ROI'] = curr_roi
        if count == 0:
            count += 1
            CV_summary_df = tmp_CV_summary_df[tmp_CV_summary_df['CV-r']>0.1]
        else:
            CV_summary_df = pd.concat([CV_summary_df, tmp_CV_summary_df[tmp_CV_summary_df['CV-r']>0.1]]).reset_index(drop=True)


# For RSA, only include images with all three valid presentations for this analysis
for curr_sub in tqdm(sub_list):
    # Load the share image list
    share_img_list = np.load(os.path.join(img_list_dir, 'sub0{}_share_img_order.npy'.format(curr_sub)))

    _bulk_size = 100
    _num_bulks = int(np.ceil(len(share_img_list)/_bulk_size))
    img_count = []
    for curr_bulk in range(_num_bulks):
        _start_ind = curr_bulk*_bulk_size
        _end_ind = (curr_bulk+1)*_bulk_size
        curr_bulk_dir = os.path.join(betas_dir,                                    'sub0{}_betas_{}to{}_share_norm'.format(curr_sub,
                                                   _start_ind,
                                                    _end_ind))
        with open(curr_bulk_dir, "rb") as f:
            curr_beta = pickle.load(f)
        curr_bulk_count = [k.shape[1] for k in curr_beta]
        img_count = img_count + curr_bulk_count

    curr_sub_count = pd.DataFrame(img_count, columns=['count'])
    curr_sub_count['img_ind'] = share_img_list
    curr_sub_count['SUB'] = curr_sub
    if curr_sub == 1:
        all_sub_count = curr_sub_count
    else:
        all_sub_count = all_sub_count.append(curr_sub_count).reset_index(drop=True)

all_sub_count_rep = all_sub_count[all_sub_count['count'] == 3]
rep_img_count = all_sub_count_rep.groupby(by='img_ind').count().reset_index()
filtered_img_count = rep_img_count[rep_img_count['SUB'] == 8]['img_ind'].values


# Try to subsample vvs to match performance
for curr_sub in tqdm(sub_list):
    print(curr_sub)
    lPFC_count = len(CV_summary_df[(CV_summary_df['SUB']==curr_sub)&
                                  (CV_summary_df['ROI']=='lPFC')])
    curr_sub_lPFC = CV_summary_df[(CV_summary_df['SUB']==curr_sub)&
                                 (CV_summary_df['ROI']=='lPFC')].reset_index(drop=True)
    curr_sub_vvs = CV_summary_df[(CV_summary_df['SUB']==curr_sub)&
                                (CV_summary_df['ROI']=='vvs')].sort_values(by='CV-r').reset_index(drop=True)
    
    # Try to match the mean
    lPFC_min = curr_sub_lPFC['CV-r'].min()
    lPFC_max = curr_sub_lPFC['CV-r'].max()
    n_lPFC = len(curr_sub_lPFC)
    filtered_vvs = curr_sub_vvs[(curr_sub_vvs['CV-r']<lPFC_max)&
                               (curr_sub_vvs['CV-r']>lPFC_min)].reset_index(drop=True)
    curr_sub_vvs_sampled = filtered_vvs.sample(n_lPFC, weights=np.ceil((len(filtered_vvs)-filtered_vvs.index.values)/1000), random_state=curr_seed)
    
    while curr_sub_vvs_sampled['CV-r'].mean()-curr_sub_lPFC['CV-r'].mean() > 0.1:
        curr_sub_vvs_sampled = filtered_vvs.sample(n_lPFC,  weights=np.ceil((len(filtered_vvs)-filtered_vvs.index.values)/1000), random_state=curr_seed)
    
    if curr_sub == 1:
        vvs_sampled = curr_sub_vvs_sampled
    else:
        vvs_sampled = vvs_sampled.append(curr_sub_vvs_sampled).reset_index(drop=True)
print('Finished subsampling...')
        
# Conduct RSA with the subsampled vertices
similarity_dict = {}
bool_mat = np.ones((len(filtered_img_count),
                   len(filtered_img_count)), dtype=bool)
triu_mat = np.triu(bool_mat, k=1)
for curr_sub in tqdm(sub_list):
    for curr_roi in roi_list:
        # Load the share image list
        share_img_list = np.load(os.path.join(img_list_dir, 'sub0{}_share_img_order.npy'.format(curr_sub)))
        assert np.all(share_img_list[np.isin(share_img_list, filtered_img_count)] == filtered_img_count)

        # Load the betas
        if curr_roi == 'lPFC':
            final_selected_verts = list(CV_summary_df[(CV_summary_df['SUB']==curr_sub)&
                                  (CV_summary_df['ROI']=='lPFC')]['VERTEX'].values)
        elif curr_roi == 'vvs':
            final_selected_verts = list(vvs_sampled[vvs_sampled['SUB']==curr_sub]['VERTEX'].values)

        _bulk_size = 100
        _num_bulks = int(np.ceil(len(share_img_list)/_bulk_size))
        for curr_bulk in range(_num_bulks):
            _start_ind = curr_bulk*_bulk_size
            _end_ind = (curr_bulk+1)*_bulk_size
            if curr_roi == 'lPFC':
                curr_bulk_dir = os.path.join(betas_dir,
                                        'sub0{}_betas_{}to{}_share_norm'.format(curr_sub,
                                                       _start_ind,
                                                        _end_ind))
            elif curr_roi == 'vvs':
                curr_bulk_dir = os.path.join(betas_dir,
                                    'sub0{}_betas_{}to{}_{}_share_norm'.format(curr_sub,
                                                                               _start_ind,
                                                                                _end_ind,
                                                                              curr_roi))
            with open(curr_bulk_dir, "rb") as f:
                curr_beta = pickle.load(f)
            sampled_vert = np.array([np.mean(k[final_selected_verts, :], axis=1) for k in curr_beta])

            if curr_bulk == 0:
                betas_array_share = sampled_vert
            else:
                betas_array_share = np.vstack((betas_array_share, sampled_vert))
            #list_betas_unique = list_betas_unique + sampled_vert

        # included images
        included_indices = np.isin(share_img_list, filtered_img_count)
        included_images_betas = betas_array_share[included_indices, :]

        # Get a similarity matrix
        similarity_mat = euclidean_distances(included_images_betas, included_images_betas)


        similarity_dict['sub{}_{}'.format(curr_sub,
                                          curr_roi)] = similarity_mat[triu_mat]

print('Finished running RSA...')
diff_after_sample = []
for curr_sub in sub_list:
    curr_sub_lPFC = CV_summary_df[(CV_summary_df['SUB']==curr_sub)&
                                  (CV_summary_df['ROI']=='lPFC')].reset_index(drop=True)
    curr_sub_vvs = vvs_sampled[(vvs_sampled['SUB']==curr_sub)].reset_index(drop=True)
    diff_after_sample.append(curr_sub_vvs['CV-r'].mean() - curr_sub_lPFC['CV-r'].mean())
    
# Across subject
sim_df = pd.DataFrame(columns=['SUB_A', 'SUB_B', 'ROI_A', 'ROI_B', 'type', 'sim'])
for curr_sub_comb in combinations(sub_list, 2):
    for curr_roi in roi_list:
    
        sim_repA = similarity_dict['sub{}_{}'.format(curr_sub_comb[0],
                                      curr_roi)]
        sim_repB = similarity_dict['sub{}_{}'.format(curr_sub_comb[1],
                                      curr_roi)]
        
        # Need to get rid of the zeros
        selected_sim_repA = sim_repA[(sim_repA!=0)&(sim_repB!=0)]
        selected_sim_repB = sim_repB[(sim_repA!=0)&(sim_repB!=0)]
        
        curr_corr, _ = stats.spearmanr(selected_sim_repA, selected_sim_repB)

        sim_df.loc[len(sim_df)] = {'SUB_A': curr_sub_comb[0], 
                                   'SUB_B': curr_sub_comb[1], 
                                   'ROI_A': curr_roi, 
                                   'ROI_B': curr_roi, 
                                   'type':'across-subject, {}'.format(curr_roi),
                                   'sim': curr_corr}
        
# Save the results
line_to_write='{},{},{},{},{},{},{},{},{},{},{}\n'.format(curr_seed,
        sim_df[sim_df['type']=='across-subject, lPFC']['sim'].mean(),
        sim_df[sim_df['type']=='across-subject, vvs']['sim'].mean(),
        diff_after_sample[0],
        diff_after_sample[1],
        diff_after_sample[2],
        diff_after_sample[3],
        diff_after_sample[4],
        diff_after_sample[5],
        diff_after_sample[6],
        diff_after_sample[7])


with open(summary_df_dir_rsa, 'a') as f:
    f.write(line_to_write)

# Run univariate with the subsampled vertices

# First load the image features
# Apply the classifier to all the images
sub_list = range(1, 9)
# First load all the unique images
for curr_sub in sub_list:
    unique_imgs_list=np.load(os.path.join(img_list_dir, 'sub0{}_unique_img_order.npy'.format(curr_sub)))
    unique_features_array = np.load(os.path.join(feat_dir,
                    'img_sub0{}_unique.npy'.format(curr_sub)))
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
                    'img_sub0{}_share.npy'.format(1)))
all_features_array = np.vstack((all_features_array, share_features_array))
print('Finished loading pre-extracted features!')

# Find out what images are missing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(os.path.join(NSD_top_dir, 'models', 'ViT-B-32.pt'))
model.eval()
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
    
# Now train the classifier for each subject and each roi
count = 0
for curr_roi in roi_list:
    for curr_sub in sub_list:
        
        # Load features
        unique_features_array = np.load(os.path.join(feat_dir,
                            'img_sub0{}_unique.npy'.format(curr_sub)))
        share_features_array = np.load(os.path.join(feat_dir,
                            'img_sub0{}_share.npy'.format(curr_sub)))
        
        # Load the betas
        if curr_roi == 'lPFC':
            final_selected_verts = list(CV_summary_df[(CV_summary_df['SUB']==curr_sub)&
                                  (CV_summary_df['ROI']=='lPFC')]['VERTEX'].values)
        elif curr_roi == 'vvs':
            final_selected_verts = list(vvs_sampled[vvs_sampled['SUB']==curr_sub]['VERTEX'].values)
        
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
            elif curr_roi == 'vvs':
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
            elif curr_roi == 'vvs':
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
        
        # Apply the classifer to the features
        shown_imgs_act = clf.predict(all_features_array)

        df1 = pd.DataFrame(np.squeeze(shown_imgs_act), columns=['activation'])
        df1['img'] = all_imgs

        missing_imgs_act = clf.predict(missing_features)
        df2 = pd.DataFrame(np.squeeze(missing_imgs_act), columns=['activation'])
        df2['img'] = missing_imgs

        df = df1.append(df2).reset_index(drop=True)
        
        df['SUB'] = curr_sub
        df['ROI'] = curr_roi
        
        if count == 0:
            count += 1
            all_summary_df = df
        else:
            all_summary_df = pd.concat([all_summary_df, df]).reset_index(drop=True)
            

# Compute the across-subject correaltion for univariatee 
corr_sum_df = pd.DataFrame(columns=['ROI', 'SUB_1', 'SUB_2', 'Corr'])
for curr_roi in roi_list:
    for sub_pair in combinations(sub_list,2):
        df_1 = all_summary_df[(all_summary_df['SUB']==sub_pair[0])&
                             (all_summary_df['ROI']==curr_roi)]
        df_2 = all_summary_df[(all_summary_df['SUB']==sub_pair[1])&
                             (all_summary_df['ROI']==curr_roi)]
        merged_df = df_1.merge(df_2, on='img')
        curr_r, _ = stats.spearmanr(merged_df['activation_x'],
                      merged_df['activation_y'])
        corr_sum_df.loc[len(corr_sum_df)] = {'ROI':curr_roi,
                                             'SUB_1':sub_pair[0], 
                                             'SUB_2':sub_pair[1], 
                                             'Corr':curr_r}
        

# Save the results
line_to_write='{},{},{},{},{},{},{},{},{},{},{}\n'.format(curr_seed,
        corr_sum_df[corr_sum_df['ROI']=='lPFC']['Corr'].mean(),
        corr_sum_df[corr_sum_df['ROI']=='vvs']['Corr'].mean(),
        diff_after_sample[0],
        diff_after_sample[1],
        diff_after_sample[2],
        diff_after_sample[3],
        diff_after_sample[4],
        diff_after_sample[5],
        diff_after_sample[6],
        diff_after_sample[7])


with open(summary_df_dir_univariate, 'a') as f:
    f.write(line_to_write)
            
        