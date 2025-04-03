This is document to explain the file management of the project : 

Each subject will have a `PathSubject` object attached to him. 

This object will contain all the paths of the required files for the analysis, this aims to simplify and make sure that paths are right. 


# `BasePath` class for file management
 
This is the base management class for the patient. This file will contain the paths for the following files : 
- `arg` file : Graph file used to compute the transform matrix to register the image from the native space to a target space
- `native`file : The file that contain the native MRI (`nii.gz` file)

<span style="color : red"> This is the base class for all the following class that will be presented below</span>

# `SubjectPath` class

This is the main class that will be used to retreive subject specific paths.

This class will be specific to each subject and will point to different file of the file tree for preprocessed pictures. Each subject will have a specific folder containing all the files.

All those files will be divided into 3 subfolders : 
- `native` : all the file processed in the native space
- `ICBM2009c` : all the file that has been registered to the `ICBM2009c` space
- `crop` : all the file that has been cropped using a specific mask

## Folder detail : 

### `native` folder 

This folder contains all the different files produced in the native space :
- `<subject>_mean_curvature.nii.gz` : The mean curvature file processed with de `VipGeometry -m mc` tool (cf. BrainVisa)
- `<subject>_thresh_mc.nii.gz` : Thresholded volume
- `<subject>_thresh_report.json` : The threshold report containing the informations about the threshold used to produce the previous file. It also contains additional informations about the threshold
- `<subject>_white_matter_native.nii.gz` : The white matter only
- `<subject>_sulci_native.nii.gz` : The sulci only

## `ICBMc2009`folder 

This folder contain all the files of the subject registered to the `ICBM2009c` space :
- `<subject>_thresh_mc_ICBM2009c.nii.gz` : The `<subject>_thresh_mc.nii.gz` registered to ICBM
- `<subject>_white_matter_ICBM2009c.nii.gz` : The `<subject>_white_matter_native.nii.gz` registered to ICBM
- `<subject>_sulci_ICBM2009c.nii.gz` : The `<subject>_sulci_native.nii.gz`  registered to ICBM
- `<subject>_transform_report.json` : The file containing the report for the transform

## `crop` folder 

This folder contains all the cropped files, all those files are registered in the `ICBM2009c`space : 
- `<subject>_cerebellum_thresh_mc.nii.gz` : File containing the cropped `<subject>_thresh_mc_ICBM2009c.nii.gz` 
- `<subject>_cerebellum_white_matter.nii.gz` : Cropped `<subject>_white_matter_ICBM2009c.nii.gz` 
- `<subject>_cerebellum_sulci.nii.gz` : Cropped `<subject>_sulci_ICBM2009c.nii.gz` 
- `<subject>_crop_report.json` : Cropping report (path of the mask used)

## Visualisation : 
```
<subject>/
├── native/
│   ├── <subject>_mean_curvature.nii.gz
│   ├── <subject>_thresh_mc.nii.gz
│   ├── <subject>_thresh_report.json
│   ├── <subject>_white_matter_native.nii.gz
│   ├── <subject>_sulci_native.nii.gz
│
├── ICBM2009c/
│   ├── <subject>_thresh_mc_ICBM2009c.nii.gz
│   ├── <subject>_white_matter_ICBM2009c.nii.gz
│   ├── <subject>_sulci_ICBM2009c.nii.gz
│   ├── <subject>_transform_report.json
│
└── crop/
    ├── <subject>_cerebellum_thresh_mc.nii.gz
    ├── <subject>_cerebellum_white_matter.nii.gz
    ├── <subject>_cerebellum_sulci.nii.gz
    ├── <subject>_crop_report.json

```


# `MaskPath` class

This class will contain all the paths related to the masks, all the saved are stored in a unique folder. 

***At the moment the files are mask of the cerebellum but if the project seems to work we should be able to try other types of crops. The `type` of the crop then defines the part of the brain that is cropped (e.g. `cerebellum`)***

- `<type>_native.nii.gz` : The file that contains the mask mapped to the native space
- `<type>_ICBM2009c.nii.gz` : The file registered to the `ICBM2009c` space 
- `<type>_mask_report.json` : Report of the settings of the transform to `ICBM2009c` space

There is the summary : 
```
<type>/
├── <type>_native.nii.gz
├── <type>_ICBM2009c.nii.gz
└── <type>_mask_report.json
```

