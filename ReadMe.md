#TransSTAM

This is the code for the paper "Joint Spatial-Temporal and Appearance Modeling with Transformer for Multiple Object Tracking".

![img.png](images/framework.png)

Paper: is still under review

## Usage - On MOT17

1. Using docker we provide

```shell
docker pull icicle314/trans_stam_env:v1.0

nvidia-docker run -it -d --name TransSTAM_work_env --ipc=host -v /ssd:/ssd -v /home:/home icicle314/trans_stam_env:v1.0 bash

docker exec -it TransSTAM_work_env bash
```

2. Clone the enter this repository
```shell
cd /root
git clone 
```

3. prepare data 
- The models can also be downloaded Baidu (code: lq3v).
- You should place the models to path /root/LPC_MOT/model_weights/ .
- Notice: we adopt the fast-reid as our reid model. However, the authors have updated their codes. In order to get the same reid features with our trained model, we also present the codes that we used here.
```shell



```



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

