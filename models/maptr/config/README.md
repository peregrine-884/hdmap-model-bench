# MapTR Config ガイド

hdmap-model-bench では MapTR 用の独自 config は管理しません。使用する config は、すべて MapTR 公式 config です。

```text
models/maptr/external/MapTR/projects/configs/maptr/
```

この README は、公式 config のどこを修正すると学習・推論・評価にどう影響するかを読むためのガイドです。default 値は `maptr_tiny_r50_24e_bevformer.py` を基準にしています。

各項目は `設定項目:` の見出しで示します。関連する config は、本文中の `参照先` テーブルに分けて記載します。

基本的には、外部から `cfg-options` などで部分的に上書きするのではなく、使用する公式 config ファイル自体を修正します。そのため、多くの項目は大本の変数を修正すれば後続の設定に自動反映されます。`参照先` は、手動で全部を書き換えるための一覧ではなく、意図通り反映されているか確認するための一覧です。

`MapTRPredictor` を使う場合も、公式 config の path をそのまま渡します。

```python
predictor = MapTRPredictor(
    config_path="models/maptr/external/MapTR/projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py",
    checkpoint_path="checkpoints/maptr/official/maptr_tiny_r50_24e_bevformer.pth",
)
```

## 基本

###### 設定項目: `_base_`

Default:

```python
_base_ = [
    "../datasets/custom_nus-3d.py",
    "../_base_/default_runtime.py",
]
```

公式 config が継承する dataset/runtime の共通設定です。通常は変更しません。

###### 設定項目: `plugin`, `plugin_dir`

Default:

```python
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
```

MapTR の custom module を読み込むための設定です。`plugin_dir` がずれると `MapTRHead` や dataset pipeline などが import できなくなります。

## Model と出力

###### 設定項目: `map_classes`

Default:

```python
map_classes = ["divider", "ped_crossing", "boundary"]
num_map_classes = len(map_classes)
```

MapTR が出力する HD map class を決めます。

参照先:

| 種別 | 参照している config |
| --- | --- |
| class 数 | `num_map_classes` |
| model head | `model.pts_bbox_head.num_classes` |
| bbox coder | `model.pts_bbox_head.bbox_coder.num_classes` |
| train dataset | `data.train.map_classes` |
| val dataset | `data.val.map_classes` |
| test dataset | `data.test.map_classes` |

###### 設定項目: `class_names`

Default:

```python
class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
```

nuScenes object class の設定です。MapTR の HD map vector 出力そのものは `map_classes` 側で決まりますが、dataset pipeline の `ObjectNameFilter` や `DefaultFormatBundle3D` で使われます。

###### 設定項目: `point_cloud_range`

Default:

```python
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
```

MapTR が予測する ego 周辺の BEV 範囲です。

参照先:

| 種別 | 参照している config |
| --- | --- |
| 基準値 | `point_cloud_range` |
| transformer encoder | `model.pts_bbox_head.transformer.encoder.pc_range` |
| bbox coder | `model.pts_bbox_head.bbox_coder.pc_range` |
| train assigner | `model.train_cfg.pts.point_cloud_range` |
| train dataset | `data.train.pc_range` |
| val dataset | `data.val.pc_range` |
| test dataset | `data.test.pc_range` |

###### 設定項目: `voxel_size`

Default:

```python
voxel_size = [0.15, 0.15, 4]
```

grid や bbox coder で使われる空間解像度です。`point_cloud_range` と一緒に扱うことが多いです。

###### 設定項目: `bev_h_`, `bev_w_`

Default:

```python
bev_h_ = 200
bev_w_ = 100
```

BEV feature の解像度です。

参照先:

| 種別 | 参照している config |
| --- | --- |
| model head 高さ | `model.pts_bbox_head.bev_h` |
| model head 幅 | `model.pts_bbox_head.bev_w` |
| positional encoding 高さ | `model.pts_bbox_head.positional_encoding.row_num_embed` |
| positional encoding 幅 | `model.pts_bbox_head.positional_encoding.col_num_embed` |
| train dataset | `data.train.bev_size` |
| val dataset | `data.val.bev_size` |
| test dataset | `data.test.bev_size` |

###### 設定項目: `fixed_ptsnum_per_gt_line`, `fixed_ptsnum_per_pred_line`

Default:

```python
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
```

1本の vector line を何点で表現するかを決めます。GT 側と prediction 側で点数を変える場合、loss と evaluation の前提が合っているか確認してください。

###### 設定項目: `input_modality`

Default:

```python
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)
```

入力 sensor 種別の設定です。camera-only モデルではこの default を使います。fusion model を使う場合は公式の fusion 用 config に合わせてください。

###### 設定項目: `model.type`

Default:

```python
model = dict(type="MapTR", ...)
```

構築する detector class です。通常は変更しません。

###### 設定項目: `model.pretrained`

Default:

```python
pretrained = dict(img="ckpts/resnet50-19c8e357.pth")
```

学習開始時の image backbone 初期化に使います。推論時に MapTR の学習済み checkpoint を load する設定とは別です。

###### 設定項目: `model.img_backbone`

Default:

```python
img_backbone = dict(type="ResNet", depth=50, ...)
```

画像 backbone です。ResNet-18 など別 backbone の公式 config を使う場合は、checkpoint と構造の対応に注意してください。

###### 設定項目: `model.pts_bbox_head`

Default の主な値:

```python
type = "MapTRHead"
num_query = 900
num_vec = 50
num_pts_per_vec = fixed_ptsnum_per_pred_line
num_classes = num_map_classes
```

Map vector を予測する head です。出力数、class 数、vector 点数に関わります。

###### 設定項目: `model.pts_bbox_head.bbox_coder`

Default の主な値:

```python
type = "MapTRNMSFreeCoder"
max_num = 50
pc_range = point_cloud_range
num_classes = num_map_classes
```

head の raw 出力を評価・可視化用の vector prediction に変換する部分です。`map_classes` や `point_cloud_range` を変えた場合はここも揃える必要があります。

###### 設定項目: `model.train_cfg`

Default の主な値:

```python
grid_size = [512, 512, 1]
out_size_factor = 4
assigner.type = "MapTRAssigner"
pts_cost.weight = 5
```

学習時の assigner / cost 設定です。推論だけでは基本的に使われません。

## Dataset path

###### 設定項目: `dataset_type`

Default:

```python
dataset_type = "CustomNuScenesLocalMapDataset"
```

MapTR 用の nuScenes local map dataset class です。

###### 設定項目: `data_root`

Default:

```python
data_root = "data/nuscenes/"
```

nuScenes format dataset の root directory です。この workspace では CARLA から変換した nuScenes dataset を次の場所に置く想定です。

```text
dataset/carla_nuscenes/nuscenes/
```

公式 config を CARLA dataset に向ける場合は、ここを workspace の dataset path に変更します。

###### 設定項目: `data.train.ann_file`

Default:

```python
ann_file = data_root + "nuscenes_infos_temporal_train.pkl"
```

学習用 annotation pickle です。

###### 設定項目: `data.val.ann_file`, `data.test.ann_file`

Default:

```python
ann_file = data_root + "nuscenes_infos_temporal_val.pkl"
```

validation/test 用 annotation pickle です。

###### 設定項目: `data.val.map_ann_file`, `data.test.map_ann_file`

Default:

```python
map_ann_file = data_root + "nuscenes_map_anns_val.json"
```

MapTR の評価で使う GT map annotation json です。

###### 設定項目: `data.samples_per_gpu`, `data.workers_per_gpu`

Default:

```python
samples_per_gpu = 4
workers_per_gpu = 4
```

dataloader の batch size と worker 数です。評価時は `data.val.samples_per_gpu=1` が指定されています。

## 学習だけに効く設定

###### 設定項目: `train_pipeline`

Default の流れ:

```text
LoadMultiViewImageFromFiles
PhotoMetricDistortionMultiViewImage
LoadAnnotations3D
ObjectRangeFilter
ObjectNameFilter
NormalizeMultiviewImage
RandomScaleImageMultiViewImage(scales=[0.5])
PadMultiViewImage(size_divisor=32)
DefaultFormatBundle3D
CustomCollect3D(keys=["gt_bboxes_3d", "gt_labels_3d", "img"])
```

学習時の画像読み込み、augmentation、正規化、padding、collect の流れです。推論時には使われません。

###### 設定項目: `optimizer`

Default:

```python
optimizer = dict(
    type="AdamW",
    lr=6e-4,
    paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.1)}),
    weight_decay=0.01,
)
```

学習 optimizer です。

###### 設定項目: `optimizer_config`

Default:

```python
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
```

gradient clipping の設定です。

###### 設定項目: `lr_config`

Default:

```python
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
```

learning rate schedule です。

###### 設定項目: `total_epochs`, `runner`

Default:

```python
total_epochs = 24
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
```

学習 epoch 数と runner の設定です。

###### 設定項目: `checkpoint_config`

Default:

```python
checkpoint_config = dict(interval=1)
```

checkpoint を何 epoch ごとに保存するかを決めます。

###### 設定項目: `log_config`

Default:

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ],
)
```

学習 log の出力間隔と hook です。

###### 設定項目: `fp16`

Default:

```python
fp16 = dict(loss_scale=512.0)
```

mixed precision 学習の設定です。

## 推論・評価に効く設定

###### 設定項目: `test_pipeline`

Default の流れ:

```text
LoadMultiViewImageFromFiles
NormalizeMultiviewImage
MultiScaleFlipAug3D
  RandomScaleImageMultiViewImage(scales=[0.5])
  PadMultiViewImage(size_divisor=32)
  DefaultFormatBundle3D(with_label=False)
  CustomCollect3D(keys=["img"])
```

validation、official test script、visualization、`MapTRPredictor` 用入力の前処理に対応する pipeline です。

###### 設定項目: `data.val`, `data.test`

Default:

```python
data.val.pipeline = test_pipeline
data.test.pipeline = test_pipeline
```

MapTR 公式の評価 script や可視化 script が使う dataset 設定です。

###### 設定項目: `evaluation`

Default:

```python
evaluation = dict(interval=2, pipeline=test_pipeline, metric="chamfer")
```

評価 metric と評価間隔です。

`MapTRPredictor` は raw output を返すだけです。score threshold による filtering、class ごとの整形、可視化用の変換は notebook や評価 script 側で扱います。

## 複数モデルを比較する場合

MapTR 公式が用意している複数モデルを比較する場合は、使用する公式 config と対応する checkpoint を切り替えます。

```text
maptr_tiny_r50_24e.py
maptr_tiny_r50_24e_bevformer.py
maptr_tiny_r50_24e_bevpool.py
maptr_tiny_r50_110e.py
maptr_nano_r18_110e.py
maptr_tiny_fusion_24e.py
```

hdmap-model-bench 側に重複した wrapper config は作らず、基本的には使用する MapTR 公式 config を直接修正・管理します。

## 修正時の注意

公式 config は、学習・推論・評価の設定が1ファイルにまとまっています。上部の大本の変数を修正すれば、多くの関連設定は後続の dict に自動反映されます。

ただし、変更後は次の参照先が意図通りの値になっているか確認してください。

- `map_classes` と `num_classes`
- `point_cloud_range` と dataset/model/bbox coder
- `bev_h_`, `bev_w_` と positional encoding/dataset `bev_size`
- `data_root` と `ann_file` / `map_ann_file`

hdmap-model-bench では、基本的に外部 override ではなく公式 config ファイル自体を編集する方針です。
