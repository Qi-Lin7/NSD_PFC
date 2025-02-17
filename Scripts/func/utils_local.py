import os
import pandas as pd
from PIL import Image
import numpy as np
import nibabel as nib
import h5py
from scipy.stats import zscore
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def load_sub_resp_df(sub, top_dir):

    resp_dir = os.path.join(top_dir,
                            'data',
                            'nsddata',
                            'ppdata',
                            'subj0{}'.format(sub),
                            'behav',
                            'responses.tsv')
    resp_df = pd.read_csv(resp_dir,
                          sep='\t')
    return resp_df


def preproc_and_extract(img_ind, data_obj, extractor, preproc_obj, layer, flatten=True):

    img_array = data_obj['imgBrick'][img_ind - 1, :, :, :]  # the 73k ID is 1-based, but python is 0-based
    img_obj = Image.fromarray(np.uint8(img_array)).convert('RGB')

    prep_img = preproc_obj(img_obj).unsqueeze(0)
    feat_array = extractor(prep_img)
    if flatten:
        feat_vect = feat_array[layer].detach().numpy().flatten()
    else:
        feat_vect = feat_array[layer].detach().numpy().squeeze()
    return feat_vect

def load_and_mask(img_ind, sub, stim_seq, lh_mask, rh_mask, betas_dir):

    curr_img_post = np.where(stim_seq[sub - 1, :] == img_ind)
    # Pregenerate an array to hold the patterns
    curr_img_patterns = np.zeros((np.sum(lh_mask) + np.sum(rh_mask),
                                  len(curr_img_post[0])))

    for ind, curr_pos in enumerate(curr_img_post[0]):
        # position is 0-based
        curr_ses = int(np.floor(curr_pos / 750)+1) # session is 1-based

        curr_ses_pos = curr_pos % 750  # pos is 0-based

        if curr_ses < 10:
            ses_str = '0{}'.format(curr_ses)
        else:
            ses_str = str(curr_ses)

        lh_beta_dir = os.path.join(betas_dir, 'lh.betas_session{}.mgh'.format(ses_str))
        lh_beta_array = nib.load(lh_beta_dir).get_fdata().squeeze()
        lh_beta_masked = lh_beta_array[lh_mask, curr_ses_pos]

        rh_beta_dir = os.path.join(betas_dir, 'rh.betas_session{}.mgh'.format(ses_str))
        rh_beta_array = nib.load(rh_beta_dir).get_fdata().squeeze()
        rh_beta_masked = rh_beta_array[rh_mask, curr_ses_pos]

        both_hemi = np.hstack((lh_beta_masked, rh_beta_masked))
        curr_img_patterns[:, ind] = both_hemi

    return curr_img_patterns


def load_and_mask_native(img_ind, sub, stim_seq, lh_mask, rh_mask, betas_dir):

    curr_img_post = np.where(stim_seq[sub - 1, :] == img_ind)
    # Pregenerate an array to hold the patterns
    if lh_mask.dtype == 'bool':
        curr_img_patterns = np.zeros((np.sum(lh_mask) + np.sum(rh_mask),
                                      len(curr_img_post[0])))
    elif lh_mask.dtype == 'int64':
        curr_img_patterns = np.zeros((len(lh_mask) + len(rh_mask),
                                      len(curr_img_post[0])))
    for ind, curr_pos in enumerate(curr_img_post[0]):
        # position is 0-based
        curr_ses = int(np.floor(curr_pos / 750)+1) # session is 1-based
        curr_ses_pos = curr_pos % 750  # pos is 0-based

        if curr_ses < 10:
            ses_str = '0{}'.format(curr_ses)
        else:
            ses_str = str(curr_ses)

        lh_beta_dir = os.path.join(betas_dir, 'lh.betas_session{}.hdf5'.format(ses_str))
        lh_beta_obj = h5py.File(lh_beta_dir, 'r')
        # z-score the data within session
        _before_zscore = lh_beta_obj['betas'][:, lh_mask].T
        print(_before_zscore.shape)
        _after_zscore = zscore(_before_zscore, axis=1)
        lh_beta_masked = _after_zscore[:, curr_ses_pos]
        lh_beta_obj.close()

        rh_beta_dir = os.path.join(betas_dir, 'rh.betas_session{}.hdf5'.format(ses_str))
        rh_beta_obj = h5py.File(rh_beta_dir, 'r')
        #rh_beta_masked = rh_beta_obj['betas'][curr_ses_pos, rh_mask].T
        # z-score the data within session
        _before_zscore = rh_beta_obj['betas'][:, rh_mask].T
        print(_before_zscore.shape)
        _after_zscore = zscore(_before_zscore, axis=1)
        rh_beta_masked = _after_zscore[:, curr_ses_pos]
        rh_beta_obj.close()

        both_hemi = np.hstack((lh_beta_masked, rh_beta_masked))
        curr_img_patterns[:, ind] = both_hemi

    return curr_img_patterns

def load_native(img_ind, sub, stim_seq, betas_dir):

    curr_img_post = np.where(stim_seq[sub - 1, :] == img_ind)
    for ind, curr_pos in enumerate(curr_img_post[0]):
        # position is 0-based
        curr_ses = int(np.floor(curr_pos / 750)+1) # session is 1-based
        curr_ses_pos = curr_pos % 750  # pos is 0-based

        if curr_ses < 10:
            ses_str = '0{}'.format(curr_ses)
        else:
            ses_str = str(curr_ses)

        lh_beta_dir = os.path.join(betas_dir, 'lh.betas_session{}.hdf5'.format(ses_str))
        lh_beta_obj = h5py.File(lh_beta_dir, 'r')
        # z-score the data within session
        _before_zscore = lh_beta_obj['betas'][:,:].T
        print(_before_zscore.shape)
        _after_zscore = zscore(_before_zscore, axis=1)
        lh_beta_masked = _after_zscore[:, curr_ses_pos]
        lh_beta_obj.close()

        rh_beta_dir = os.path.join(betas_dir, 'rh.betas_session{}.hdf5'.format(ses_str))
        rh_beta_obj = h5py.File(rh_beta_dir, 'r')
        #rh_beta_masked = rh_beta_obj['betas'][curr_ses_pos, rh_mask].T
        # z-score the data within session
        _before_zscore = rh_beta_obj['betas'][:,:].T
        print(_before_zscore.shape)
        _after_zscore = zscore(_before_zscore, axis=1)
        rh_beta_masked = _after_zscore[:, curr_ses_pos]
        rh_beta_obj.close()
        if ind == 0:
            curr_img_patterns = np.zeros((lh_beta_masked.shape[0] + rh_beta_masked.shape[0],
                                      len(curr_img_post[0])))

        both_hemi = np.hstack((lh_beta_masked, rh_beta_masked))
        curr_img_patterns[:, ind] = both_hemi

    return curr_img_patterns

def load_and_mask_ses_native(ses, sub, lh_mask, rh_mask, betas_dir):
    
    if ses < 10:
        ses_str = '0{}'.format(int(ses))
    else:
        ses_str = str(int(ses))
            
    lh_beta_dir = os.path.join(betas_dir, 'lh.betas_session{}.hdf5'.format(ses_str))
    lh_beta_obj = h5py.File(lh_beta_dir, 'r')
    # z-score the data within session
    _before_zscore = lh_beta_obj['betas'][:, lh_mask].T
    print(_before_zscore.shape)
    _after_zscore = zscore(_before_zscore, axis=1)
    lh_beta_masked = _after_zscore
    lh_beta_obj.close()

    rh_beta_dir = os.path.join(betas_dir, 'rh.betas_session{}.hdf5'.format(ses_str))
    rh_beta_obj = h5py.File(rh_beta_dir, 'r')
    #rh_beta_masked = rh_beta_obj['betas'][curr_ses_pos, rh_mask].T
    # z-score the data within session
    _before_zscore = rh_beta_obj['betas'][:, rh_mask].T
    print(_before_zscore.shape)
    _after_zscore = zscore(_before_zscore, axis=1)
    rh_beta_masked = _after_zscore
    rh_beta_obj.close()

    both_hemi = np.vstack((lh_beta_masked, rh_beta_masked))
    
    return both_hemi

def load_and_mask_native_memory(img_ind, sub, stim_seq, resp_df, vert_ind, hemi , betas_dir):

    curr_img_post = np.where(stim_seq[sub - 1, :] == img_ind)
    #print(len(curr_img_post[0]))
    if len(curr_img_post[0])>1:
        # Pregenerate an array to hold the patterns
        curr_img_resp = np.zeros((len(curr_img_post[0])-1, 1))
        curr_img_delay = np.zeros((len(curr_img_post[0])-1, 1))
        curr_img_patterns = np.zeros((len(curr_img_post[0])-1, 1))
        count = 0
        for ind, curr_pos in enumerate(curr_img_post[0]):
            # position is 0-based
            curr_ses = int(np.floor(curr_pos / 750)+1) # session is 1-based
            curr_ses_pos = curr_pos % 750  # pos is 0-based
            
            curr_ses_df = resp_df[resp_df['SESSION']==curr_ses].sort_values(by=['RUN', 'TRIAL']).reset_index(drop=True)
            assert len(curr_ses_df) == 750
            curr_trial_info = curr_ses_df.iloc[curr_ses_pos]
            assert curr_trial_info['73KID']==img_ind
            
            if curr_trial_info['ISOLD']== 1:
                if curr_ses < 10:
                    ses_str = '0{}'.format(curr_ses)
                else:
                    ses_str = str(curr_ses)

                lh_beta_dir = os.path.join(betas_dir, '{}.betas_session{}.hdf5'.format(hemi,
                                                                                       ses_str))
                lh_beta_obj = h5py.File(lh_beta_dir, 'r')
                lh_beta_masked = lh_beta_obj['betas'][curr_ses_pos, vert_ind].T
                lh_beta_obj.close()

                curr_img_patterns[count] = lh_beta_masked

                curr_img_resp[count] = curr_trial_info['ISCORRECT']
                curr_img_delay[count]= curr_trial_info['MEMORYRECENT']
                count += 1
        return curr_img_patterns, curr_img_resp, curr_img_delay
    else:
        return None, None, None


def load_memory_var(img_ind, sub, stim_seq, resp_df):
    curr_img_post = np.where(stim_seq[sub - 1, :] == img_ind)
    # print(len(curr_img_post[0]))
    if len(curr_img_post[0]) > 1:
        # Pregenerate an array to hold the patterns
        curr_img_resp = np.zeros((len(curr_img_post[0]) - 1, 1))
        curr_img_delay = np.zeros((len(curr_img_post[0]) - 1, 1))
        count = 0
        for ind, curr_pos in enumerate(curr_img_post[0]):
            # position is 0-based
            curr_ses = int(np.floor(curr_pos / 750) + 1)  # session is 1-based
            curr_ses_pos = curr_pos % 750  # pos is 0-based

            curr_ses_df = resp_df[resp_df['SESSION'] == curr_ses].sort_values(by=['RUN', 'TRIAL']).reset_index(
                drop=True)
            assert len(curr_ses_df) == 750
            curr_trial_info = curr_ses_df.iloc[curr_ses_pos]
            assert curr_trial_info['73KID'] == img_ind

            if curr_trial_info['ISOLD'] == 1:

                curr_img_resp[count] = curr_trial_info['ISCORRECT']
                curr_img_delay[count] = curr_trial_info['MEMORYRECENT']
                count += 1
        return True, curr_img_resp, curr_img_delay
    else:
        return False, None, None
    
def load_memory_var_encoding(img_ind, sub, stim_seq, resp_df):
    curr_img_post = np.where(stim_seq[sub - 1, :] == img_ind)
    # print(len(curr_img_post[0]))
    if len(curr_img_post[0]) > 1:
        # Pregenerate an array to hold the encoding pattern 
        # We only consider the first repetition
        curr_img_resp = np.zeros((1, 1))
        curr_img_delay = np.zeros((1, 1))
        count = 0
        for ind, curr_pos in enumerate(curr_img_post[0]):
            # position is 0-based
            curr_ses = int(np.floor(curr_pos / 750) + 1)  # session is 1-based
            curr_ses_pos = curr_pos % 750  # pos is 0-based

            curr_ses_df = resp_df[resp_df['SESSION'] == curr_ses].sort_values(by=['RUN', 'TRIAL']).reset_index(
                drop=True)
            assert len(curr_ses_df) == 750
            curr_trial_info = curr_ses_df.iloc[curr_ses_pos]
            assert curr_trial_info['73KID'] == img_ind

            if (curr_trial_info['ISOLD'] == 1) & (count == 0):
                curr_img_resp[count] = curr_trial_info['ISCORRECT']
                curr_img_delay[count] = curr_trial_info['MEMORYRECENT']
                count += 1
        return True, curr_img_resp, curr_img_delay
    else:
        return False, None, None


def reorder_betas_feats_train(betas, feats, train_inds):
    
    num_patterns = [betas[k].shape[1] for k in list(train_inds)]
    total_num_patterns = np.sum(num_patterns)
    
    # Pregenerate holders
    train_betas_array = np.zeros((total_num_patterns, betas[0].shape[0]))
    train_features_array = np.zeros((total_num_patterns, feats.shape[1]))
    
    count = 0
    for ind, curr_train_ind in enumerate(list(train_inds)):
        # ind refers to the actual index of each element in the train_inds, 
        # curr_train_ind refers to the element itself (i.e., indexes list_betas and features array)
        
        curr_num_patterns = num_patterns[ind]
        train_betas_array[count:count+curr_num_patterns, :] = betas[curr_train_ind].T
        
        _feat_array = np.repeat(np.expand_dims(feats[curr_train_ind,:], -1),
                               curr_num_patterns, axis=1)
        train_features_array[count:count+curr_num_patterns, :] = _feat_array.T
        count += curr_num_patterns
    
    return train_betas_array, train_features_array

def reorder_betas_feats_test(betas, feats, test_inds):
    
    test_betas_array = np.array([np.nanmean(betas[k], axis=1) if betas[k].shape[1]>1 else np.squeeze(betas[k]) for k in list(test_inds)])
    test_feats_array = feats[test_inds, :]
    
    return test_betas_array, test_feats_array
