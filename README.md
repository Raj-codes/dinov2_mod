# DINOv2_Mod

A custom Python package for image feature extraction, logistic regression training, and prediction using DINOv2.

## Folder Structure
- **dinov2_mod/func**: Custom functions for the pipeline.
- **dinov2_mod/pretrained_weights**: Pretrained weights for DINOv2.
- **dinov2_mod/supplementary**: Supporting files like `label_map.json`.logistic_regression_model.
- **dinov2_mod/tests**: Test scripts.

## Pretrained weights

## Pretrained models finetuned on NCT-CRC-100K

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th># of<br />iterations</th>
      <th>CRC-VAL-HE-7K<br />20-NN balanced acc</th>
      <th>CRC-VAL-HE-7K<br />linear balanced acc</th>
      <th>teacher backbone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14</td>
      <td align="right">21 M</td>
      <td align="right">2k</td>
      <td align="right">93.8%</td>
      <td align="right">92.7%</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_NCT_10k_training_1999_teacher_checkpoint.pth?download=1">teacher weights</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="right">10k</td>
      <td align="right">93.4%</td>
      <td align="right">93.7%</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_NCT_training_9999_teacher_checkpoint.pth?download=1">teacher weights</a></td>
    </tr>
  </tbody>
</table>

## Pretrained models finetuned on TCGA

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th># of<br />iterations</th>
      <th>TCGA<br />AUROC</th>
      <th>CPTAC<br />AUROC</th>
      <th>teacher backbone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14</td>
      <td align="right">21 M</td>
      <td align="right">30k</td>
      <td align="right">89%</td>
      <td align="right">85%</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_TCGA_training_29999_teacher_checkpoint.pth?download=1">teacher weights</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="right">60k</td>
      <td align="right">84%</td>
      <td align="right">79%</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_teacher_checkpoint.pth?download=1">teacher weights</a></td>
    </tr>
  </tbody>
</table>


## Installation
1. Clone the Repository

```bash
git clone https://github.com/Raj-codes/dinov2_mod.git
cd dinov2_mod
```

2. Set Up a Python Environment

```bash
conda create -n dinov2mod python=3.9 -y
conda activate dinov2mod
```

3. Install Dependencies

A. To install all dependencies for training and evaluation(expects Linux environment):

```bash
pip install -r requirements.txt
```

B. To only run the pretrained model

```bash
pip install -r requirements_eval_windows.txt
```

4. Install the dinov2_mod package using setup.py

```bash
pip install -e .
```

## Pipeline run

1. Activate environment and open to directory where dinov2_mod folder is located

```bash
conda activate dinov2mod
cd path/to/dinov2_mod/folder
```
2. Configure the config.json file inside the dinov2_mod directory

```json
{
  "wsi_path": "/path/to/your/wsi/file.tif",
  "patch_size": 224,
  "output_dir": "/path/to/output/folder",
  "model_type": "vits14",
  "model_checkpoint": "/path/to/dinov2_model_weights.pth",
  "predict_model_path": "/path/to/logistic_regression_model.joblib",
  "label_map_path": "/path/to/label_map.json",
  "output_json_path": "/path/to/output_predictions.json",
  "model_batch_size": 128
}
```
Arguments:

--wsi_path: The path to the WSI image file.

--patch_size: Size of the image patches (default: 224).

--output_dir: Directory where the image tiles will be saved (default: outputs).

--model_type: The type of the DINOv2 model (vits14 or vitg14) (default: vits14).

--model_checkpoint: The path to the DINOv2 pretrained weights file.

--predict_model_path: The path to the logistic regression model file.

--label_map_path: The path to the label map JSON file.

--output_json_path: The output path for saving the embeddings and predictions in JSON format.

--model_batch_size: number of images used in one batch by the model


3. Running __main__.py

```bash
python -m dinov2_mod --config_path "dinov2_mod/config.json"
```


## Output

After running the command, the system will:

- Crop the WSI image into patches.
- Extract features from the image tiles using DINOv2.
- Predict the labels for each tile using the logistic regression model.
- Save the results (embeddings and predictions) in the specified JSON file.


```json
[
    {
        "image_path": "path/to/tile1.png",
        "predicted_label": "Label1",
        "embedding": [0.1, 0.2, 0.3, ...]
    },
    {
        "image_path": "path/to/tile2.png",
        "predicted_label": "Label2",
        "embedding": [0.4, 0.5, 0.6, ...]
    }
]
```


## Notes

The model used for feature extraction is DINOv2, which requires CUDA for GPU-based acceleration. Ensure you have an appropriate CUDA-enabled GPU setup if running on a machine with a GPU.
The output file will contain both the predicted labels and the embeddings for each image tile, which can be used for downstream tasks.