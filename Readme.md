# AICITY2024_Track5
This repo includes training and testing code of our work, An Effective Method for Detecting Violation of Helmet Rule for Motorcyclists, which ranks #2 in [AICtity2024](https://www.aicitychallenge.org/2024-challenge-tracks/) Track5.

## Installation
1. DETA

Please follow instructions from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) for installation, data preparation, and additional usage examples.

2. Co-DETR

Please follow instructions from [Co-DETR](https://github.com/Sense-X/Co-DETR) for installation, data preparation, and additional usage examples.

## Data preparation

1. extract frames
```bash
./video2img.sh video2img './data/videos' './data/frames'
```
The format of dataset will be saved as follows:
>   - data
>     - videos
>     - ReadMe.txt
>     - gt.txt 
>     - lables.txt  
>     - frames
>       - 001
>           - 000001.jpg
>           - ...
>           - 000200.jpg
>       - 002
>       - ...
>       - 100

2. generate coco labels
```bash
python generate_label.py
python aicity2coco.py
```

## Train
1. train DETA

The pretrained checkpoint of DETA can be downloaded from [model-O365](https://utexas.box.com/s/5jgu0nfzdcln4b6eknwz981q0kzgv36l). After downloading the checkpoint, please put the files into ./helmet/DETA/weights/

Now you can train your DETA model with the following command:
```bash
cd helmet/DETA
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/deta_swin_ft_alldata.sh --finetune ./weights/adet_swin_pt_o365.pth
```
2. train Co-DETR

Please modify the config file `projects/configs/codino_swinl_V1_trainall.py` before training Co-DETR. 
Please change `data_root`, `ann_file` and `img_prefix` in this config file to the location where you put your coco-format data. 

The pretrained checkpoint of Co-DETR can be downloaded from [model-o365tococo](https://drive.google.com/drive/folders/1r7bTh-DXkkQsCqkHYAHv4U-2hoEsLEoM). After downloading the checkpoint, please put it under './pretrained' or modify the `load_from` field in your config file to the path of your downloaded checkpoint. 

Now you can train your Co-DETR model with the following command:
```bash
cd helmet/Co-DETR
bash scripts/dist_train.sh
```
After training, the checkpoints will be saved to `work_dir`. You can change this path by modifying the last line of `scripts/dist_train.sh`.   

## Inference
1. inference with DETA
```bash
cd helmet/DETA
python main.py --eval --resume $MODEL_WEIGHT --with_box_refine --two_stage --num_feature_levels 5 --num_queries 900 --dim_feedforward 2048 --dropout 0.0 --cls_loss_coef 1.0 --assign_first_stage --assign_second_stage --epochs 24 --lr_drop 20 --lr 5e-5 --lr_backbone 5e-6 --batch_size 1 --backbone swin --bigger --submit --output_dir $OUTPUT_DIR
```

2. format the DETA output

Replace the input/output path  in the python file.
```bash
python format_deta_result.py
```

3. inference with Co-DETR

Please modify `scripts/infer.py`, change `config_file`, `checkpoint_file`, `out_dir`, `img_dir` in `scripts/infer.py` accordingly.
Then you can inference with Co-DETR model with the following command:
```bash
cd helmet/Co-DETR
python scripts/infer.py
```
The detection result will be saved to `out_dir`. Each image will generate a json file under `{out_dir}/preds`.

4. format the Co-DETR output

To merge the results of seperate json files into one txt file, please modify `json_dir` and `save_txt_path` in `scripts/format_result.py`, then run the following command:
```bash
python scripts/format_result.py
```

5. model ensemble

Modify `pred_path_xxx` in `model_ensemble.py` to the path of your DETA output and Co-DETR output, then run the following script:
```bash
python model_ensemble.py
```
The final result for submission will be saved to `save_txt_path`.

## Result visualization
Replace the frames path and results txt path in the python file.
```bash
python visible.py
```