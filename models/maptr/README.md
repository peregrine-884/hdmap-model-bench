# MapTR

このディレクトリは、hdmap-model-bench から vendored official MapTR repository を扱うための integration です。

## 方針

MapTR の config は公式 config を直接使います。

```text
models/maptr/external/MapTR/projects/configs/maptr/
models/maptr/external/MapTR/projects/configs/maptrv2/
```

hdmap-model-bench 側では、MapTR 用の wrapper config は管理しません。公式 config の読み方・修正方針は次にまとめています。

```text
models/maptr/config/README.md
```

## ファイル構成

```text
interface/
  predictor.py
  types.py
  input_adapters.py
  runtime.py
```

MapTR 固有の Python interface 実装は `interface/` にまとめています。notebook や scripts からは `models.maptr.interface.predictor` などの import path を使います。

```text
interface/predictor.py
```

MapTR model を build し、checkpoint を load し、MapTR 形式の 1 sample 入力を受け取ります。`predict()` は model-bench/eval 側で扱いやすい `MapTRPrediction` を返します。公式 MapTR の raw output を確認したい debug 用途では `predict_raw()` を使い、`format_output()` で型付き prediction に変換します。

`interface/predictor.py` が行わないこと:

- 共通 schema への変換
- metric 計算
- prediction の可視化

```text
interface/types.py
```

MapTR interface の入力・出力の型を定義します。`CameraSensor`, `MapTRSensorInput`, `MapVector`, `MapTRPrediction` をここに置き、`interface/predictor.py` はこれらを受け取って 1 回分の推論を行うだけにします。

```text
interface/input_adapters.py
```

`hdmap-driving-eval` の共通 schema を MapTR 入力へ変換する adapter です。現在は `CarlaSensorSample -> MapTRSensorInput` を扱います。CARLA 側で用意する sensor packet は `hdmap-driving-eval/schemas/carla_sensor_sample.py` で定義し、MapTR 固有の camera order / 座標変換 / `lidar2img` / `can_bus` 整形はこの file に閉じ込めます。

```text
interface/runtime.py
```

vendored MapTR package の import と、公式 source tree を直接変更しないための runtime compatibility patch をまとめます。

```text
utils/visualization.py
```

MapTR / MapTRv2 の raw prediction output を BEV 上に描画する helper です。公式 output の次の形を想定しています。

```python
raw[0]["pts_bbox"]["boxes_3d"]
raw[0]["pts_bbox"]["scores_3d"]
raw[0]["pts_bbox"]["labels_3d"]
raw[0]["pts_bbox"]["pts_3d"]
```

```text
notebooks/01_nuscenes_baseline.ipynb
```

公式 nuScenes の 1 sample で MapTR の正規入力、`MapTRSensorInput` 再構成結果、pipeline 後の構造、raw output、簡易 BEV 可視化を確認する notebook です。

```text
notebooks/02_carla_nuscenes_baseline.ipynb
```

CARLA から export した nuScenes 形式 data で、公式 MapTR dataset / pipeline の baseline を確認する notebook です。

```text
notebooks/03_carla_adapter.ipynb
```

`carla_raw_data` の 1 frame から `CarlaSensorSample` を作り、`interface/input_adapters.py` で MapTR 入力へ変換して、MapTR pipeline / inference / BEV 可視化まで確認する notebook です。

```text
scripts/train.py
scripts/evaluate.py
```

公式 MapTR の `tools/train.py` と `tools/test.py` を呼び出す薄い launcher です。vendored MapTR checkout を cwd / PYTHONPATH に設定し、引数は公式 script へ渡します。

```text
../../../../scripts/setup_maptr_env.sh
```

MapTR の公式実行環境を作る bash script です。conda env、PyTorch、MMCV、mmdet/mmdet3d、`nuscenes-devkit` などをまとめて install します。

```text
scripts/prepare_data.py
```

公式 MapTR の `tools/maptrv2/custom_nusc_map_converter.py` を呼び出し、MapTR 用の nuScenes annotation pkl を生成する launcher です。`nuscenes` と `carla_nuscenes` を interactive に切り替えられます。

CARLA nuScenes は version 名と map 名が公式 nuScenes と異なるため、公式 converter を直接編集せず、一時ファイル上で map 名と split 処理だけを合わせて実行します。

## Data flow

### 01 nuScenes baseline

公式 nuScenes と公式 MapTR dataset pipeline を基準にして、MapTR の正規入力、正解 annotation、推論出力を確認します。

Input data:

| 変数 | 内容 |
| --- | --- |
| `CONFIG_PATH` | 公式 MapTR config |
| `CHECKPOINT_PATH` | 公式 MapTR checkpoint |
| `NUSCENES_ROOT` | nuScenes dataset root |
| `ANN_FILE` | 公式 converter が生成した `nuscenes_map_infos_temporal_{train,val}.pkl` |
| `SAMPLE_INDEX` | `ANN_FILE` から確認する 1 sample の index |

Intermediate data:

| 変数 | 作り方 | 役割 |
| --- | --- | --- |
| `raw_info` | `dataset.data_infos[SAMPLE_INDEX]` | pkl に保存された raw sample dict。正解 vector は `raw_info['annotation']` に入る |
| `official_info` | `dataset.get_data_info(SAMPLE_INDEX)` | MapTR dataset class が作る公式入力 dict |
| `gt_vectors` | `raw_info['annotation']` | 公式変換済み pkl に含まれる local vectorized map annotation |
| `rebuilt_input` | `MapTRSensorInput(...)` | `interface/types.py` の形式に詰め直した入力 object |
| `rebuilt_info` | `rebuilt_input.to_maptr_info()` | `interface/types.py` 形式から MapTR dict に戻した入力 |
| `example` | `dataset.prepare_test_data(SAMPLE_INDEX)` | 公式 test pipeline 後の model 入力 |

Output data:

| 変数 / file | 内容 |
| --- | --- |
| `raw_output` | 公式 MapTR model の生出力 |
| `prediction` | `MapTRPrediction` に整形した出力 |
| GT BEV plot | `gt_vectors` の正解 map 可視化 |
| GT + prediction BEV plot | 正解を点線、推論を実線で重ねた簡易比較 |
| `outputs/maptr/nusc/<sample_token>.json` | `prediction.to_dict()` を保存した JSON |

Processing flow:

```text
ANN_FILE pkl
  -> dataset
  -> raw_info / official_info
  -> gt_vectors (raw_info['annotation'])
       -> rebuilt_input / rebuilt_info  # interface/types.py 形式との contract 比較
       -> example                       # 公式 pipeline 後の model 入力
            -> raw_output
            -> prediction
            -> outputs/maptr/nusc/<sample_token>.json
```

重要: 推論に使うのは `example`、つまり公式 dataset pipeline から作った入力です。`rebuilt_input` は contract 比較用です。

### 02 CARLA adapter

CARLA raw data を `CarlaSensorSample` にまとめ、`interface/input_adapters.py` で MapTR 入力へ変換できるか確認します。

Input data:

| 変数 / file | 内容 |
| --- | --- |
| `RUN_DIR` | 1 run / repetition の CARLA raw data directory |
| `FRAME_ID` | 確認する frame id。例: `0000` |
| `metadata/<FRAME_ID>.json.gz` | frame の sensor file path、timestamp、map name、channel 一覧 |
| `ego/<FRAME_ID>.json.gz` | ego pose、can_bus、velocity、acceleration、IMU 系の値 |
| `SENSOR_CONFIG_PATH` | sensor attachment / calibration / camera intrinsic |
| `samples/<CAMERA>/<FRAME_ID>.jpg` | 6 camera image |
| `samples/LIDAR_TOP/<FRAME_ID>.laz` | lidar payload path。adapter contract では path と pose を確認する |

Intermediate data:

| 変数 | 作り方 | 役割 |
| --- | --- | --- |
| `metadata` | `metadata/<FRAME_ID>.json.gz` を read | raw frame metadata |
| `ego_payload` | `ego/<FRAME_ID>.json.gz` を read | ego pose と motion source |
| `sensor_config` | `vehicle_nissan_micra.json` を read | camera/lidar extrinsic と intrinsic source |
| `carla_sample` | raw metadata + ego + sensor config | `CarlaSensorSample` schema の入力 object |
| `maptr_input` | `maptr_input_from_carla(carla_sample)` | adapter 後の `MapTRSensorInput` object |
| `maptr_info` | `maptr_input.to_maptr_info()` | MapTR pipeline に渡す dict 形式 |

Output / checks:

| 出力 | 内容 |
| --- | --- |
| contract check | `maptr_info` が 01 baseline と同じ key / shape を持つか |
| numeric print | `can_bus`, `lidar2ego`, `lidar2img` の sanity check |
| image grid | raw camera images の channel order 確認 |
| projection smoke check | `lidar2ego` / `lidar2img` の向きが大きく破綻していないか |
| pipeline / inference check | `maptr_input` を MapTR pipeline に通し、`raw_output` / `prediction` を生成できるか |

Processing flow:

```text
carla_raw_data RUN_DIR + FRAME_ID
  -> metadata / ego_payload / sensor_config
  -> carla_sample                  # hdmap-driving-eval schema
  -> maptr_input                   # interface/input_adapters.py output
  -> maptr_info                    # MapTR input dict
       -> contract check
       -> image order check
       -> projection smoke check
       -> MapTR inference
       -> raw_output / prediction
       -> outputs/maptr/carla_adapter/<sample_id>.json
```

重要: `03_carla_adapter.ipynb` では `carla_raw_data` を主入力にします。export 済みの `carla_nuscenes/nuscenes` は主入力ではなく、必要になった場合の reference 比較用です。

### Downstream planner boundary

`models/maptr` が直接扱う接続点は、CARLA 側から MapTR interface に渡す入力と、MapTR から出る `MapTRPrediction` までです。

```text
CARLA -> MapTR interface -> MapTR -> MapTRPrediction
```

MapTR output から planner input への変換、planner tick との対応付け、class mapping、planner 座標系への変換は `hdmap-driving-eval` 側の責務です。

CARLA 側から MapTR interface に必要な data:

| 必須度 | CARLA / raw source | `CarlaSensorSample` field | 用途 |
| --- | --- | --- | --- |
| required | sample id / frame id | `sample_id` | 入力と MapTR 出力を対応付ける id |
| required | timestamp | `timestamp` | sensor packet の時刻 |
| required | ego vehicle pose in CARLA world | `ego_pose.matrix` or `ego_pose.translation` + `rotation_quat` | CARLA world -> MapTR/nuScenes ego frame 変換 |
| required | 6 camera image paths | `cameras[*].image_path` | MapTR image input |
| required | 6 camera intrinsics | `cameras[*].intrinsic` | projection / image geometry |
| required | 6 camera sensor-to-world poses | `cameras[*].sensor_to_world` | `camera2ego`, `lidar2img` の生成 |
| strongly recommended | top lidar sensor-to-world pose | `lidar.sensor_to_world` | MapTR の lidar/ego BEV frame を安定させる |
| optional | lidar payload path | `lidar.point_path` | MapTR wrapper の `pts_filename`。画像-only 推論でも path として保持 |
| strongly recommended | ego velocity | `ego_motion.velocity` | `can_bus` |
| strongly recommended | ego acceleration | `ego_motion.acceleration` | `can_bus` |
| strongly recommended | ego angular velocity / IMU gyro | `ego_motion.angular_velocity` or `imu.angular_velocity` | `can_bus` |
| optional | map name / route context | `map_context` / `metadata` | debug、downstream 側の対応付け |

CARLA -> MapTR を接続する前に確認すること:

- 全 camera image が同一 timestamp の packet として揃っている
- ego pose と sensor pose が同じ timestamp / frame convention である
- MapTR interface の出力が 01 baseline の shape / key contract と一致する
- MapTR 推論結果が BEV 上で極端に反転・スケールずれしていない


## Checkpoint 配置

checkpoint は `hdmap-model-bench` の中には置かず、workspace root の `checkpoints/maptr/` で管理します。

```text
checkpoints/maptr/official/
checkpoints/maptr/trained/
checkpoints/maptr/experiments/
```

つまり、この README を workspace root から見た場合の配置は次の形です。

```text
hdmap-driving-workspace/
  checkpoints/maptr/
  repos/hdmap-model-bench/
```

公式 config が `ckpts/resnet50-19c8e357.pth` のように vendored MapTR checkout 内の path を参照している場合は、workspace 側の checkpoint path に修正します。

例:

```python
from pathlib import Path
import importlib.util
import sys

predictor_path = Path("repos/hdmap-model-bench/models/maptr/interface/predictor.py")
spec = importlib.util.spec_from_file_location("maptr_predictor", predictor_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

predictor = module.MapTRPredictor(
    config_path="repos/hdmap-model-bench/models/maptr/external/MapTR/projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py",
    checkpoint_path="checkpoints/maptr/official/maptr_tiny_r50_24e_bevformer.pth",
)
```

## 公式 training の起動

先に MapTR 実行環境を作ります。

```bash
bash scripts/setup_maptr_env.sh
conda activate maptr
```

デフォルトは公式手順に合わせて `Python 3.8`, `torch==1.9.1+cu111`, `mmcv-full==1.4.0` を使います。

`mmdetection3d` の CUDA extension build で `/usr/include/c++/11/... std_function.h` のようなエラーが出る場合は、system GCC 11 を使っていることが原因になりやすいです。workspace の `scripts/setup_maptr_env.sh` はデフォルトで conda env 内に GCC/G++ 9 を入れ、`CC`, `CXX`, `CUDAHOSTCXX` をその compiler に向けて build します。

`cuda_runtime.h` / `cuda_runtime_api.h` が見つからない場合は、CUDA toolkit の include path が build に渡っていません。workspace の `scripts/setup_maptr_env.sh` は `nvcc` が `PATH` から見えることを前提にしています。

既存環境を使う場合は、`gcc` / `g++` が CUDA と互換のある version を向いているか確認してください。Ubuntu 22.04 の system GCC 11 は古い MapTR / mmdet3d / CUDA extension build では失敗することがあります。

環境作成後、training は次のように起動します。

```bash
python repos/hdmap-model-bench/models/maptr/scripts/train.py \
  repos/hdmap-model-bench/models/maptr/external/MapTR/projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py \
  --work-dir outputs/maptr/train/maptr_tiny_r50_24e_bevformer
```

追加引数は公式 `tools/train.py` にそのまま渡されます。

## MapTR 用 nuScenes data preparation

公式 converter は `nuscenes-devkit` と MapTR/MMCV 系の実行環境を必要とします。今回のように `No module named 'nuscenes'` が出る場合は、実行している Python 環境に `nuscenes-devkit` を入れてください。

```bash
python -m pip install nuscenes-devkit
```

`prepare_data.py` は実行前に import check を行い、不足している package を表示します。command だけ確認する `--dry-run` では依存関係 check は行いません。

interactive に対象 dataset を選ぶ場合:

```bash
python repos/hdmap-model-bench/models/maptr/scripts/prepare_data.py
```

nuScenes を指定して実行する場合:

```bash
python repos/hdmap-model-bench/models/maptr/scripts/prepare_data.py \
  --dataset nuscenes \
  --version v1.0-trainval
```

nuScenes mini を指定して実行する場合:

```bash
python repos/hdmap-model-bench/models/maptr/scripts/prepare_data.py \
  --dataset nuscenes \
  --version v1.0-mini
```

CARLA nuScenes を指定して実行する場合:

```bash
python repos/hdmap-model-bench/models/maptr/scripts/prepare_data.py \
  --dataset carla_nuscenes
```

実行前にコマンドだけ確認する場合:

```bash
python repos/hdmap-model-bench/models/maptr/scripts/prepare_data.py \
  --dataset carla_nuscenes \
  --dry-run
```

デフォルトでは次の場所を使います。

```text
nuscenes:
  root_path: dataset/nuscenes/nuscenes
  out_dir:   dataset/nuscenes/nuscenes
  canbus:    dataset/nuscenes
  version:   v1.0-trainval
  extra_tag: nuscenes

carla_nuscenes:
  root_path: dataset/carla_nuscenes/nuscenes/nuscenes
  out_dir:   dataset/carla_nuscenes/nuscenes/nuscenes
  canbus:    dataset/carla_nuscenes/nuscenes
  version:   v1.0-carla
  extra_tag: carla_nuscenes
```

公式 converter は `*_map_infos_temporal_train.pkl` / `*_map_infos_temporal_val.pkl` を生成します。通常の `*_infos_temporal_train.pkl` とは別物で、MapTR 用の local vector map annotation を含みます。

## 公式 evaluation の起動

```bash
python repos/hdmap-model-bench/models/maptr/scripts/evaluate.py \
  repos/hdmap-model-bench/models/maptr/external/MapTR/projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py \
  checkpoints/maptr/official/maptr_tiny_r50_24e_bevformer.pth \
  --eval chamfer
```

追加引数は公式 `tools/test.py` にそのまま渡されます。
