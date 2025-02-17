import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import os, sys, time, pickle, configparser

exp_config = configparser.ConfigParser()
exp_config.read('../config')
NSD_top_dir = exp_config['DIR']['NSD_top_dir']
sub_list = range(1, 9)
modality = 'img'

# Load masks
mask_output_dir = os.path.join(NSD_top_dir, 'intermediate', 'masks', 'HCP_MMP_ROIs')

# Results directory
results_dir = os.path.join(NSD_top_dir, 'results', 'RSS_CLIP')
output_dir = os.path.join(NSD_top_dir, 'results', 'CLIP_selected_verts')

for curr_sub in sub_list:
    results_df_dir = os.path.join(results_dir, 'sub0{}_{}_RSS_norm.csv'.format(curr_sub, modality))
    results_df = pd.read_csv(results_df_dir, header=None, names=['Vertex', 'RSS', 'NaNs']).sort_values(by='RSS').reset_index(drop=True)
    lh_frontal_mask = np.load(os.path.join(mask_output_dir, 'lh.lPFC.sub0{}.npy'.format(curr_sub)))
    rh_frontal_mask = np.load(os.path.join(mask_output_dir, 'rh.lPFC.sub0{}.npy'.format(curr_sub)))
    total_vertices = np.sum(lh_frontal_mask) + np.sum(rh_frontal_mask)

    assert len(results_df) == total_vertices
    selected_df = results_df.iloc[-int(total_vertices*0.1):]
    selected_df['SUB'] = curr_sub
    if curr_sub == 1:
        all_selected_df = selected_df
    else:
        all_selected_df = pd.concat([all_selected_df, selected_df]).reset_index(drop=True)

all_selected_df.to_csv(os.path.join(output_dir, '{}_lPFC_selected_verts.csv'.format(modality)))


