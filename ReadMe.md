#TransSTAM

This is the code for the paper "Joint Spatial-Temporal and Appearance Modeling with Transformer for Multiple Object Tracking".

![img.png](images/framework.png)

Paper: is still under review

## Usage - On MOT17

1. Clone the enter this repository
```commandline
git clone 
```


2. Create a docker image for this project:


3. 


### prepare data
```
# download MOT dataset
```


### To train model

```
bash scripts/train.sh "TransSTAM" trans_stam 2 ${Train.hdf5} ${To_inference_pbs}
```

### To ablation model

```
# build datasets for ablation study
bash scripts/build_3folder_cross_dataset.sh ${Root_dir}

# Do ablation study
bash scripts/ablation.sh trans_stam ${tag} ${layer_num} ${cache_window} ${sample_window} ${aspe} ${rstpe} ${ape} ${ape_style} ${TRACKRESDIR} ${MOT17DIR} ${EVALUATIONRESDIR}

# example
bash scripts/ablation.sh trans_stam "baseline" 2 20 50 "with_abs_pe" "with_relative_pe" "with_assignment_pe" "diff" ${TRACKRESDIR} ${MOT17DIR} ${EVALUATIONRESDIR}

```

