# Individual differences in prefrontal coding of visual features

This repo contains the code base for running the main analyses presented in the following manuscript:
Lin, Q., & Lau, H. (2024). Individual differences in prefrontal coding of visual features. bioRxiv. 

## Setup and download
### 0.1 Clone the Repository
```
git clone ttps://github.com/Qi-Lin7/NSD_PFC.git
cd NSD_PFC
```

### 0.2 Install the conda environment
```
conda env create -f environment.yml
conda activate GenPFC_mini
```
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



