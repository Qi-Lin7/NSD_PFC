import nibabel as nib
from nilearn.signal import clean
from nilearn.masking import apply_mask
from nilearn import maskers
from nilearn.signal import clean
import numpy as np
import pandas as pd
import os, sys, configparser
exp_config = configparser.ConfigParser()
exp_config.read('../config')
NSD_top_dir = exp_config['DIR']['NSD_top_dir']
sys.path.append(os.path.join(NSD_top_dir,
                             'Scripts',
                             'nsdcode',
                             'nsdcode'))
from nsd_mapdata import NSDmapdata
from nsd_datalocation import nsd_datalocation
from nsd_output import nsd_write_fs
from utils import makeimagestack

base_path = os.path.join(NSD_top_dir, 'data')
# initiate NSDmapdata
nsd = NSDmapdata(base_path)
nsd_dir = nsd_datalocation(base_path=base_path)
nsd_betas = nsd_datalocation(base_path=base_path, dir0='betas')

timeseries_vol_dir = os.path.join(base_path,'nsddata_timeseries','ppdata')
output_dir = os.path.join(NSD_top_dir,'intermediate', 'resting_state')
anat_mask_dir = os.path.join(NSD_top_dir,'intermediate','masks', 'anat_masks')
nsd_anat_dir = os.path.join(NSD_top_dir, 'data', 'nsddata', 'ppdata')
motion_filtered = pd.read_csv(os.path.join(NSD_top_dir, 'results', 'motion_filtered.csv'))

curr_row_ind = int(int(sys.argv[1])-1)
frames_to_ignore = 4
curr_row = motion_filtered.iloc[curr_row_ind]

curr_sub = curr_row['SUB']
curr_file = curr_row['file name']
curr_ses = curr_row['SESSION']
curr_run = curr_row['RUN']
curr_sub_vol_dir = os.path.join(timeseries_vol_dir, 'subj0{}'.format(curr_sub),'func1pt8mm','timeseries')
curr_vol_file_dir = os.path.join(curr_sub_vol_dir, curr_file)
curr_vol = nib.load(curr_vol_file_dir)

# Extract global signals, WM and ventricles signals
small_brain_mask_dir = os.path.join(anat_mask_dir, 'subj0{}_brain_mask.nii.gz'.format(curr_sub))
small_masked_signals = apply_mask(curr_vol, small_brain_mask_dir)
gloabl_signal = np.expand_dims(np.mean(small_masked_signals, axis=1), axis=-1)

large_brain_mask_dir = os.path.join(nsd_anat_dir,
                                    'subj0{}'.format(curr_sub),
                                    'func1pt8mm',
                                    'brainmask.nii.gz')
large_masked_signals = apply_mask(curr_vol, large_brain_mask_dir)

vent_mask_dir = os.path.join(anat_mask_dir, 'subj0{}_vent_mask.nii.gz'.format(curr_sub))
curr_vent_signal = apply_mask(curr_vol, vent_mask_dir)
vent_signal = np.expand_dims(np.mean(curr_vent_signal , axis=1), axis=-1)


wm_mask_dir = os.path.join(anat_mask_dir, 'subj0{}_wm_mask.nii.gz'.format(curr_sub))
curr_wm_signal =  apply_mask(curr_vol, wm_mask_dir)
wm_signal = np.expand_dims(np.mean(curr_wm_signal, axis=1), axis=-1)

print('Finished loading masks...')

# Read the current motion file
curr_motion_file = os.path.join(timeseries_vol_dir, 'subj0{}'.format(curr_sub),
                                   'func1pt8mm','motion','motion_{}_{}.tsv'.format(curr_file.split('_')[1],
                                                                               curr_file.split('_')[2].split('.')[0]))
curr_motion_df = pd.read_csv(curr_motion_file,
                            header=None, sep='\t',
                            names=['trans_AP',
                                  'trans_LR',
                                  'trans_SI',
                                  'roll',
                                  'pitch',
                                  'yaw'])
curr_motion_diff = curr_motion_df.diff().fillna(0)
print('Finished loading motion...')

confounds_array = np.hstack((gloabl_signal[frames_to_ignore:, :],
                            np.diff(gloabl_signal, axis=0)[frames_to_ignore-1:, :],
                            vent_signal[frames_to_ignore:, :],
                             np.diff(vent_signal, axis=0)[frames_to_ignore-1:, :],
                            wm_signal[frames_to_ignore:, :],
                              np.diff(wm_signal, axis=0)[frames_to_ignore-1:, :],
                             curr_motion_df.values[frames_to_ignore:, :],
                             curr_motion_diff.values[frames_to_ignore:, :]
                            ))

# Preprocess the timeseries
cleaned_signals = clean(large_masked_signals[frames_to_ignore:, :], confounds=confounds_array,
                       detrend=True, standardize='zscore', standardize_confounds=True,
                        filter='butterworth', low_pass=0.08, high_pass=0.009, t_r=1.33,
                        ensure_finite=False)
print('Finished running the signal cleaning...')

# Save the preprocessed data into a volume
brain_masker = maskers.NiftiMasker(mask_img=large_brain_mask_dir)
brain_masker.fit()
invert_data = brain_masker.inverse_transform(cleaned_signals)
tmp_work_dir = os.path.join(NSD_top_dir, 'tmp_work')

vol_output_dir = os.path.join(tmp_work_dir, 'preproc_sub0{}_ses{}_run{}.nii.gz'.format(curr_sub,
                                                                                curr_ses,
                                                                                curr_run))
nib.save(invert_data, vol_output_dir)

print('Finished saving the results...')


# Resample the data into fsnative space
sourcedata = vol_output_dir
hemi_list = ['lh', 'rh']
curr_sub_output = os.path.join(output_dir,
                               'subj0{}'.format(curr_sub))
for curr_hemi in hemi_list:
    data = []
    for p in range(3):
        data.append(
            nsd.fit(
                curr_sub,
                'func1pt8',
                f'{curr_hemi}.layerB{p+1}',
                sourcedata,
                'cubic',
                badval=0
            )
        )
    data = np.asarray(data)
    mean_ts = np.mean(data, axis=0)
    output_file_dir = os.path.join(curr_sub_output, '{}.session{}_run{}.npy'.format(curr_hemi,
                                                                                   curr_ses,
                                                                                   curr_run))
    np.save(output_file_dir, mean_ts)









