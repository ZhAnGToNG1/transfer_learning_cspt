## Pre-training phase

The pre-training includes two steps: 
* **Pre-training on ImageNet-1K**: For step (1),the pre-trained models ([ViT-B](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth), [ViT-L](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large_full.pth)) from [MAE](https://github.com/facebookresearch/mae) are adopted.


* **Continual pre-training on remote sesning datasets**: For step (2), to pretrain ViT-B, run the following commond:
``` 
python transfer_learning_dspt/main_pretrain.py
        --batch_size 64
        --model mae_vit_base_patch16
        --epochs 1600
        --blr 1.5e-4
        --weight_decay 0.05
        --norm_pix_loss
        --data_path  ./your_data_path
        --output_dir ./pretrain/your_save_path
        --log_dir  ./pretrain/your_save_path
        --resume ./pretrained_model_path

```
## Fine-tuning phase
### Scene Classification task:
To fine-tune the pre-trained models on classificaiton tasks, you can run the following commond:
```
python transfer_learning_dspt\fine-tuning\classification\train_finetune.py
      --accum_iter 4
      --batch_size 32
      --model  vit_base_patch16
      --finetune ./your_pretrained_model_path
      --epochs 100
      --blr 5e-4
      --layer_decay 0.65
      --weight_decay 0.05
      --drop_path 0.1
      --reprob 0.25
      --mixup 0.8
      --cutmix 1.0
      --dist_eval
      --data_path ./your_data_path
      --output_dir ./finetune/your_save_path
      --log_dir ./finetune/your_save_path
```

### Land cover classification (Segmentation) task:
```
python transfer_learning_dspt\fine-tuning\segmentation\tools\train.py
../configs/dspt/upernet/upernet_dspt_base_12_512_slide_gid.py
--work-dir
./output_dir/your_save_path
--seed
0

Note: modify your pretrained model path i.e.,mim_model in the corresponding config file.
```



### Object detection task:
```
python transfer_learning_dspt\fine-tuning\detection\tools\train.py
../configs/mask_rcnn/mask_rcnn_vit_base_fpn_1x_ucas_aod.py
--work-dir
./output_dir/your_save_path
--seed
0

Note: modify your pretrained model path i.e.,mim_model in the corresponding config file.
```
