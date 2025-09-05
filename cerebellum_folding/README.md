This is document to explain the file management of the project : 

Each subject will have a `PathSubject` object attached to him. 

This object will contain all the paths of the required files for the analysis, this aims to simplify and make sure that paths are right. 


# `BasePath` class for file management
 
This is the base management class for the patient. This file will contain the paths for the following files : 
- if available : `transform_mat` : The path for the `trm` file used for the registration
- `arg` file : Graph file used to compute the transform matrix to register the image from the native space to a target space (not necessary if transformation matrix is not available
- `native`file : The file that contain the native MRI (`nii.gz` file)

<span style="color : red"> This is the base class for all the following class that will be presented below</span>

# `SubjectPath` class

This object aims to be able to easily use the data sourced from different folder and to standardize the way data used in this project is stored. It is fully customizable and can be extended to any database available.

This is the main class that will be used to retreive subject specific paths.

This class will be specific to each subject and will point to different file of the file tree for preprocessed pictures. Each subject will have a specific folder containing all the files.

All those files will be divided into 3 subfolders : 
- `ICBM2009c` : all the file that has been registered to the `ICBM2009c` space
- `masked` : all the file that has been masked using a specific mask
- `cropped` : File that have been cropped (bounding box around the mask)

## Folders detail : 

## `ICBMc2009`folder 

This folder contain all the files of the subject registered to the `ICBM2009c` space :
- `<subject>_resampled_icbm.nii.gz` : The raw t1-weighted mri registered in `ICBM2009c` space and resampled using cubic interpolation 
- `<subject>_mean_curvature_icbm.nii.gz` : The `<subject>_resampled_icbm.nii.gz` where mean curvature has bee n applied. 
- `<subject>_thresh_mc.nii.gz` : The `<subject>_mean_curvature_icbm.nii.gz` that has been thresholded

## `masked` folder 

This folder is subdivided in multiple subfolder where each one corresponds to a type of mask (cerebellum, vermis, ...) applied on the different files registered to the ICBM space. Each subfolder contains those files :
- `<subject>_masked_t1mri_<type>.nii.gz` : The mask that has been applied on the t1mri image
- `<subject>_masked_mean_curvature_<type>.nii.gz` : The mask that has been applied on the mean curvature image
- `<subject>_masked_thresh_<type>.nii.gz` : The mask that has been applied on the thresholded mean curvature image

## Visualisation : 
```
<subject>/
├── ICBM2009c/
│   ├── <subject>_resampled_icbm.nii.gz
│   ├── <subject>_mean_curvature_icbm.nii.gz
│   ├── <subject>_thresh_mc.nii.gz
│
├── masked/
│   ├── <type_1>/
│   |   ├── <subject>_masked_thresh_<type_1>.nii.gz
│   |   ├── <subject>_masked_mean_curvature_<type_1>.nii.gz
│   |   ├── <subject>_masked_t1mri_<type_1>.nii.gz
│   ├── <type_2>/
│   |   ├── <subject>_masked_thresh_<type_2>.nii.gz
│   |   ├── <subject>_masked_mean_curvature_<type_2>.nii.gz
│   |   ├── <subject>_masked_t1mri_<type_2>.nii.gz
│   ├── <type_3>/
│       ├── <subject>_masked_thresh_<type_3>.nii.gz
│       ├── <subject>_masked_mean_curvature_<type_3>.nii.gz
│       ├── <subject>_masked_t1mri_<type_3>.nii.gz
├── cropped/
    ├── <type_1>/
    |   ├── <subject>_crop_thresh_<type_1>.nii.gz
    |   ├── <subject>_crop_mean_curvature_<type_1>.nii.gz
    |   ├── <subject>_crop_t1mri_<type_1>.nii.gz
    ├── <type_2>/
    |   ├── <subject>_crop_thresh_<type_2>.nii.gz
    |   ├── <subject>_crop_mean_curvature_<type_2>.nii.gz
    |   ├── <subject>_crop_t1mri_<type_2>.nii.gz
    ├── <type_3>/
        ├── <subject>_crop_thresh_<type_3>.nii.gz
        ├── <subject>_crop_mean_curvature_<type_3>.nii.gz
        ├── <subject>_crop_t1mri_<type_3>.nii.gz

```


# `MaskPath` class

This class will contain all the paths related to the masks, all the saved are stored in a unique folder. 

***At the moment the files are mask of the cerebellum and the vermis but if the project seems to work we should be able to try other types of crops. The `type` of the crop then defines the part of the brain that is cropped (e.g. `cerebellum`)***

- `<type>_native.nii.gz` : The file that contains the mask mapped to the native space
- `<type>_ICBM2009c.nii.gz` : The file registered to the `ICBM2009c` space 

There is the summary : 
```
<type>/
├── <type>_native.nii.gz
├── <type>_ICBM2009c.nii.gz
└── <type>_mask_report.json
```

## Additional classes :

For each dataset, it is possible to retrieve a template of the different SubjectPath for the dataset : some parameters are already filled. 
It is possible to retrieve the path for the following datasets : 
- AtaxiaSubject
- UkbSubject
- ABCDSubject

