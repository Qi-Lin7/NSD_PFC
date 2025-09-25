# Individual differences in prefrontal coding of visual features

This repo contains the code base for running the main analyses presented in the following manuscript:
Lin, Q., & Lau, H. (2024). Individual differences in prefrontal coding of visual features. bioRxiv. 

## Setup and download
### 0.1 Clone the Repository
```
git clone https://github.com/Qi-Lin7/NSD_PFC.git
cd NSD_PFC
```

### 0.2 Install the conda environment
```
conda env create -f environment.yml
conda activate GenPFC_mini
```
This should take about 1 hour. All the analyses were run in Python 3.7.12. For the specific versions of all the packages, refer to the environment.yml file. 
### 0.3 Download the data from NSD (http://naturalscenesdataset.org)
The downloaded data should be placed under the ./data directory. 
We need:
1. Single-trial betas in the native surface space:./nsddata_betas/ppdata/subj0X/nativesurface/betas_fithrf_GLMdenoise_RR/
2. The experiment info (e.g., trial structure and stimuli order etc.): ./nsddata/experiments/nsd/
3. Freesurfer output: ./nsddata/freesurfer/
4. Resting state timeseries and motion info: ./nsddata_timeseries/ppdata/subj0X/func1pt8mm/timeseries/timeseries_sessionXX_run01.nii.gz
./nsddata_timeseries/ppdata/subj0X/func1pt8mm/motion/motion_sessionXX_run01.nii.gz
5. Stimuli: ./nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5

### 0.4 Install nsdcode
We additionally need some functionality provided by the NSD team. 
```
cd Scripts
git clone https://github.com/cvnlab/nsdcode.git
```
## Preparation
In the ./Scripts/prepration directory, there are a few scripts useful for generating the various ROI masks and inspecting the motion files from resting state. The lPFC masks and image lists have been generated so running Define lPFC.ipynb and Get_img_list.ipynb is not necessary. To generate the timeseries of visual ROIs and the dataframe containing the resting state runs included for all subjects after exclusion based on motion, run Define visual ROIs.ipynb and Map_Resting_State.ipynb. The output can be found in ./intermediate/grouped_visual_ROIs/ and ./results/

## Analyses
### 1.1 Figure 1: Building and evaluating encoding models of LPFC
Note that because the size of NSD, this part should ideally be run on a high-performance cluster. 
#### Step 1: Extract the CLIP features and LPFC betas
To extract betas in LPFC, use ./Scripts/func/BetasExt_unique.py (for extracting betas corresponding to the subject-specific images, used for training) and ./Scripts/func/BetasExt_share.py (for extracting betas corresponding to the share images, used for validation). Use ./Scripts/submit_script/submit_beta.sh to submit individual jobs that extract betas corresponding to bulks of 100 images in size (each job should take about 1 hour). The resulting output can be found in ./intermediate
/betas_native/

To extract CLIP features, use ./Scripts/func/Extract_CLIP.py. Use ./Scripts/submit_script/submit_feature.sh to submit individual jobs that extract features for each subject. The resulting output can be found in ./intermediate/features/CLIP_features

#### Step 2: Run vertex-wise ridge regression relaitng CLIP features to beta timeseries for vertex selection in the training set
Use ./Scripts/func/RidgeR_noCV_CLIP.py and ./Scripts/submit_script/submit_ridge.sh to submit individual jobs that run the ridge regression on bulks of 500 vertices. The resulting output should be in ./results/RSS_CLIP. 

After finishing the ridge regression in the training set for all subjects, run
```
python ./Scripts/func/select_verts.py
```
This generates a dataframe containing the vertex indices with a R2 in the top 10% of all the LPFC vertices based on the ridge regressin in the training image set. The resulting ouptut should be in ./results/CLIP_selected_verts. 

#### Step 3: Run cross-validation in the selected vertices
Use ./Scripts/func/RidgeR_CV_CLIP.py and ./Scripts/submit_script/submit_ridge.sh to submit individual jobs that run the ridge regression and validation on bulks of 300 vertices. The resulting output should be in ./results/CV_results_CLIP. CSV files corresponding to the results plotted in the paper have been uploaded in case the user only wants to reproduce the graphs without needing to run the entire analysis.  

#### Step 4: Plot the results
Run ./Scripts/check_results/Figure1_Check_regression_results.ipynb to reproduce the results panels in Figure 1. 

### 1.2 Figure 2: Functional connectivity
#### Step 1: Preprocess and map the resting state data to native surface space
Use ./Scripts/func/preproc_rs.py and ./Scripts/submit_script/submit_rs.sh to submit individual jobs that preprocess and map resting state data for individual runs. The resulting output should be in ./intermediate/resting_state/.

#### Step 2: Run the analysis and plot the results
Run ./Scripts/check_results/Figure2_Functional_connectivity.ipynb to reproduce the results panels in Figure 2.

### 1.3 Figure 3: RSA
#### Step 1: Extract the ventral visual stream betas
To extract betas in ventral visual stream (and A1, for Figure S6), use ./Scripts/func/BetasExt_unique_control.py (for extracting betas corresponding to the subject-specific images, used for training) and ./Scripts/func/BetasExt_share_control.py (for extracting betas corresponding to the share images, used for validation). Use ./Scripts/submit_script/submit_beta.sh to submit individual jobs that extract betas corresponding to bulks of 100 images in size (each job should take about 1 hour). The resulting output can be found in ./intermediate
/betas_native/

#### Step 2: Run the cross-validation in ventral visual stream vertices
Use ./Scripts/func/RidgeR_CV_CLIP.py and ./Scripts/submit_script/submit_ridge.sh to submit individual jobs that run the ridge regression and validation on bulks of 300 vertices. The resulting output should be in ./results/CV_results_CLIP. CSV files corresponding to the results plotted in the paper have been uploaded in case the user only wants to reproduce the graphs without needing to run the entire analysis.   

#### Step 3: Run the analysis and plot the results
Run ./Scripts/check_results/Figure3_RSA.ipynb to reproduce the results panels in Figure 3.

### 1.4 Figure 4: Univariate activation
#### Step 1: Train and run the predictive models
Use ./Scripts/func/CLIP_ForwardModel_univariate_reg.py and ./Scripts/submit_script/submit_univariate.sh to submit individual jobs that train an encoding model of the average activation for each ROI (LPFC or visual regions) and generate predicted activation for all 730000 images. The results output should be in be in ./results/Pred_activation.

#### Step 2: Plot results
Run ./Scripts/check_results/Figure4_Univariate.ipynb to reproduce the results panels in Figure 4.

### 1.5 Figure 5: Computational modeling
Codes used for this analysis are modified based on https://github.com/florapython/FlexibleWorkingMemory

## Citation
If you use the code from this repo, please cite
```
@article{lin2024individual,
  title={Individual differences in prefrontal coding of visual features},
  author={Lin, Qi and Lau, Hakwan},
  journal={bioRxiv},
  pages={2024--05},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```





