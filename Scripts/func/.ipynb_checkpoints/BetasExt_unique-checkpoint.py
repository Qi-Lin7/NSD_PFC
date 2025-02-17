import nibabel as nib
import numpy as np
import os, time, sys, pickle, configparser
import scipy.io as sio
import pandas as pd
from joblib import Parallel, delayed

from utils_local import load_sub_resp_df, load_and_mask_native


exp_config = configparser.ConfigParser()
exp_config.read('../config')
NSD_top_dir = exp_config['DIR']['NSD_top_dir']
comb_df = pd.read_csv(os.path.join(NSD_top_dir,
                      'Scripts',
                      'submit_script',
                      'array_info',
                      'Beta_Extract.csv'))
curr_row = int(sys.argv[1])-1
curr_sub = int(comb_df.iloc[curr_row]['SUB'])
curr_bulk = int(comb_df.iloc[curr_row]['bulk'])

size_bulk = int(exp_config['PARAM']['beta_bulk_size'])
start_ind = curr_bulk*size_bulk
end_ind = (curr_bulk+1)*size_bulk
print(start_ind)
print(end_ind)

mask_output_dir = os.path.join(NSD_top_dir, 'intermediate', 'masks', 'HCP_MMP_ROIs')
intermediate_dir = os.path.join(NSD_top_dir, 'intermediate', 'features')
betas_dir = os.path.join(NSD_top_dir,
                         'data',
                        'nsddata_betas',
                        'ppdata',
                        'subj0{}'.format(curr_sub),
                        'nativesurface',
                        'betas_fithrf_GLMdenoise_RR')


# Load masks
lh_frontal_mask = np.load(os.path.join(mask_output_dir, 'lh.lPFC.sub0{}.npy'.format(curr_sub)))
rh_frontal_mask = np.load(os.path.join(mask_output_dir, 'rh.lPFC.sub0{}.npy'.format(curr_sub)))

# Load exp design and stim info
exp_design = sio.loadmat(os.path.join(NSD_top_dir,
                                      'data',
                                     'nsddata',
                                     'experiments',
                                     'nsd',
                                     'nsd_expdesign.mat'))
subjectim = exp_design['subjectim']
masterordering = exp_design['masterordering']
shared_imgs = exp_design['sharedix'][0]
all_seq = subjectim[:, masterordering-1].squeeze()
curr_sub_df = load_sub_resp_df(curr_sub, NSD_top_dir)
unique_sessions = list(curr_sub_df['SESSION'].unique())
unique_sessions.sort()
curr_sub_df_filtered = curr_sub_df[curr_sub_df['SESSION'].isin(unique_sessions)]
unique_imgs_list = np.load(os.path.join(intermediate_dir, 'sub0{}_unique_img_order.npy'.format(curr_sub)))

# Load the betas
if end_ind < len(unique_imgs_list):
    betas_array = Parallel(n_jobs=10)(
        delayed(load_and_mask_native)(i, curr_sub, all_seq[:, :len(curr_sub_df_filtered)],
                              lh_frontal_mask,
                              rh_frontal_mask,
                              betas_dir) for i in unique_imgs_list[start_ind:end_ind])
else:
    betas_array = Parallel(n_jobs=10)(
        delayed(load_and_mask_native)(i, curr_sub, all_seq[:, :len(curr_sub_df_filtered)],
                                      lh_frontal_mask,
                                      rh_frontal_mask,
                                      betas_dir) for i in unique_imgs_list[start_ind:])

reordered_betas_dir = os.path.join(NSD_top_dir, 
             'intermediate',
             'betas_native',
             'sub0{}_betas_{}to{}_norm'.format(curr_sub,
                                              start_ind,
                                              end_ind))

with open(reordered_betas_dir, 'wb') as f:
    pickle.dump(betas_array, f)



