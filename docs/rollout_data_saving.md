# Rollout Data Saving 機能の実装ガイド

IsaacGym環境のシミュレーション中に、各エージェントのトルク・関節角度・ルート位置/姿勢を `.npy` ファイルとして保存する仕組みの再現手順。

---

## 概要

| 項目 | 内容 |
|------|------|
| 保存先 | `output/HumanoidIm/{exp_name}/rollout_data/` |
| 保存形式 | NumPy `.npy` (1ファイル = 1エピソード × 1環境 × 1データ種別) |
| 保存タイミング | エピソード終了（環境リセット）時 |
| 収集タイミング | 毎ステップの `post_physics_step` |
| 有効条件 | テストモード (`flags.test and flags.im_eval`) |

## アーキテクチャ

```
post_physics_step()          ← 毎ステップ呼ばれる
  └─ _track_recipient_torques()  ← バッファにデータ追加

_reset_envs(env_ids)         ← エピソード終了時に呼ばれる
  └─ _save_torque_data_for_reset_envs(env_ids)  ← バッファ → .npy保存
```

---

## Step 1: バッファの初期化

環境クラスの `__init__`（または `create_env` 等の初期化メソッド）で、空のリストを作成する。

```python
# __init__ 内
self.recipient_torque_buffer = []  # エピソード中のデータを蓄積するリスト
```

---

## Step 2: 毎ステップのデータ収集

`post_physics_step()` から呼ばれるデータ収集メソッドを実装する。テスト時のみ有効にする。

### 呼び出し側 (`post_physics_step`)

```python
from phc.utils.flags import flags

def post_physics_step(self):
    super().post_physics_step()

    # テスト＆評価モードの場合のみ収集
    if flags.test and flags.im_eval:
        self._track_recipient_torques()
```

### データ収集メソッド本体

```python
def _track_recipient_torques(self):
    """毎ステップ呼ばれ、全環境のトルク・関節角度・ルート位置/姿勢をバッファに追加"""
    all_env_ids = torch.arange(self.num_envs, device=self.device)

    if len(all_env_ids) == 0:
        return

    # --- GPU → CPU に一括転送 ---
    all_torques = self.dof_force_tensor[all_env_ids].cpu().numpy()        # [num_envs, num_dofs]
    all_joint_angles = self._dof_pos[all_env_ids].cpu().numpy()           # [num_envs, num_dofs]
    all_root_positions = self._rigid_body_pos[all_env_ids, 0, :3].cpu().numpy()  # [num_envs, 3]
    all_root_rotations = self._rigid_body_rot[all_env_ids, 0].cpu().numpy()      # [num_envs, 4] (quaternion)

    # クォータニオン → 6D回転表現への変換
    from phc.utils.torch_utils import quat_to_tan_norm
    all_root_rotations_6d = quat_to_tan_norm(
        torch.from_numpy(all_root_rotations)
    ).numpy()  # [num_envs, 6]

    # --- 環境ごとにバッファに追加 ---
    for i, env_id in enumerate(all_env_ids):
        # ロール判定（偶数=caregiver, 奇数=recipient など、タスクに応じて変更）
        role = "recipient" if env_id.item() % 2 == 1 else "caregiver"

        # モーションID・キーの取得（motion_lib がある場合）
        current_motion_id = None
        motion_key = None
        if hasattr(self._motion_lib, '_curr_motion_ids') and self._motion_lib._curr_motion_ids is not None:
            if env_id.item() < len(self._motion_lib._curr_motion_ids):
                current_motion_id = self._motion_lib._curr_motion_ids[env_id.item()].item()
                if hasattr(self._motion_lib, '_motion_data_keys') and current_motion_id < len(self._motion_lib._motion_data_keys):
                    motion_key = self._motion_lib._motion_data_keys[current_motion_id]

        self.recipient_torque_buffer.append({
            'env_id': env_id.item(),
            'role': role,
            'step': self.progress_buf[env_id].item(),
            'torques': all_torques[i].copy(),                       # [num_dofs]
            'joint_angles': all_joint_angles[i].copy(),             # [num_dofs]
            'root_position': all_root_positions[i].copy(),          # [3]
            'root_orientation_quat': all_root_rotations[i].copy(),  # [4]
            'root_orientation_6d': all_root_rotations_6d[i].copy(), # [6]
            'motion_id': current_motion_id,
            'motion_key': motion_key,
        })
```

### 収集されるデータの詳細

| フィールド | Shape (per step) | 取得元テンソル | 説明 |
|---|---|---|---|
| `torques` | `[num_dofs]` | `self.dof_force_tensor` | 各関節に加わるトルク |
| `joint_angles` | `[num_dofs]` | `self._dof_pos` | 各関節の角度 (rad) |
| `root_position` | `[3]` | `self._rigid_body_pos[:, 0, :3]` | ルートボディのXYZ位置 |
| `root_orientation_quat` | `[4]` | `self._rigid_body_rot[:, 0]` | ルートの姿勢（クォータニオン w,x,y,z） |
| `root_orientation_6d` | `[6]` | `quat_to_tan_norm()` で変換 | ルートの姿勢（6D回転表現） |

---

## Step 3: エピソード終了時の保存

`_reset_envs()` でリセット対象の環境データを保存する。

### 呼び出し側 (`_reset_envs`)

```python
def _reset_envs(self, env_ids):
    # リセット前にデータを保存
    if flags.test and flags.im_eval:
        self._save_torque_data_for_reset_envs(env_ids)

    super()._reset_envs(env_ids)
```

> **注意**: 現在のコードベースではこの呼び出しがコメントアウトされている。有効にするには `_reset_envs` 内のコメントを解除する。

### 保存メソッド本体

```python
def _save_torque_data_for_reset_envs(self, env_ids):
    """リセット対象の環境のバッファデータを .npy ファイルとして保存"""
    for env_id in env_ids:
        if len(self.recipient_torque_buffer) == 0:
            continue

        # この環境のデータだけ抽出
        env_data = [d for d in self.recipient_torque_buffer if d['env_id'] == env_id.item()]

        if len(env_data) == 0:
            continue

        # 短すぎるエピソードはスキップ（ノイズ回避）
        if len(env_data) <= 5:
            # バッファからこの環境のデータを削除して次へ
            self.recipient_torque_buffer = [d for d in self.recipient_torque_buffer if d['env_id'] != env_id.item()]
            continue

        # --- dict のリスト → NumPy 配列に変換 ---
        torque_timeseries = np.array([d['torques'] for d in env_data])                       # [T, num_dofs]
        joint_angle_timeseries = np.array([d['joint_angles'] for d in env_data])             # [T, num_dofs]
        root_position_timeseries = np.array([d['root_position'] for d in env_data])          # [T, 3]
        root_orientation_quat_timeseries = np.array([d['root_orientation_quat'] for d in env_data])  # [T, 4]
        root_orientation_6d_timeseries = np.array([d['root_orientation_6d'] for d in env_data])      # [T, 6]

        # --- 出力ディレクトリ作成 ---
        import os, time
        exp_name = getattr(self.cfg, 'exp_name', 'default_exp')
        output_dir = f"output/HumanoidIm/{exp_name}/rollout_data"
        os.makedirs(output_dir, exist_ok=True)

        # --- ユニークなファイル名を生成 ---
        motion_filename = self._get_motion_filename_from_data(env_data[0], env_id)
        role = env_data[0]['role']
        motion_id = env_data[0].get('motion_id', 'unknown')
        timestamp = int(time.time() * 1000) % 100000  # ミリ秒の下5桁

        base_filename = f"{motion_filename}_env{env_id.item()}_{role}_motion{motion_id}_{timestamp}"

        # --- 5種類のデータを個別に保存 ---
        np.save(os.path.join(output_dir, f"{base_filename}_torques.npy"), torque_timeseries)
        np.save(os.path.join(output_dir, f"{base_filename}_joint_angles.npy"), joint_angle_timeseries)
        np.save(os.path.join(output_dir, f"{base_filename}_root_positions.npy"), root_position_timeseries)
        np.save(os.path.join(output_dir, f"{base_filename}_root_orientations_quat.npy"), root_orientation_quat_timeseries)
        np.save(os.path.join(output_dir, f"{base_filename}_root_orientations_6d.npy"), root_orientation_6d_timeseries)

        print(f"Saved {role} data for env {env_id.item()}:")
        print(f"  Torques:              shape {torque_timeseries.shape}")
        print(f"  Joint angles:         shape {joint_angle_timeseries.shape}")
        print(f"  Root positions:       shape {root_position_timeseries.shape}")
        print(f"  Root orientations (quat): shape {root_orientation_quat_timeseries.shape}")
        print(f"  Root orientations (6D):   shape {root_orientation_6d_timeseries.shape}")

        # バッファからこの環境のデータを削除
        self.recipient_torque_buffer = [d for d in self.recipient_torque_buffer if d['env_id'] != env_id.item()]
```

---

## Step 4: モーションファイル名の解決

保存ファイル名にモーション名を含めるためのヘルパーメソッド。

```python
def _get_motion_filename_from_data(self, sample_data, env_id):
    """バッファのデータからファイル名用の文字列を生成"""
    import os
    if sample_data.get('motion_key') is not None:
        motion_key = sample_data['motion_key']
        if isinstance(motion_key, str):
            motion_filename = os.path.splitext(os.path.basename(motion_key))[0]
            return f"env_{env_id.item()}_{motion_filename}"
        else:
            return f"env_{env_id.item()}_motion_{motion_key}"
    elif sample_data.get('motion_id') is not None:
        return f"env_{env_id.item()}_motion_{sample_data['motion_id']}"
    else:
        return f"env_{env_id.item()}_motion"
```

---

## 出力ファイルの構造

### ディレクトリレイアウト

```
output/HumanoidIm/{exp_name}/rollout_data/
├── env_0_walkMotion001_env0_caregiver_motion3_45123_torques.npy
├── env_0_walkMotion001_env0_caregiver_motion3_45123_joint_angles.npy
├── env_0_walkMotion001_env0_caregiver_motion3_45123_root_positions.npy
├── env_0_walkMotion001_env0_caregiver_motion3_45123_root_orientations_quat.npy
├── env_0_walkMotion001_env0_caregiver_motion3_45123_root_orientations_6d.npy
├── env_1_walkMotion001_env1_recipient_motion3_45123_torques.npy
├── ...
```

### ファイル名の構成

```
{motion_filename}_env{env_id}_{role}_motion{motion_id}_{timestamp}_{data_type}.npy
```

| 要素 | 例 | 説明 |
|---|---|---|
| `motion_filename` | `env_0_walkMotion001` | `_get_motion_filename_from_data` の戻り値 |
| `env_id` | `0` | 環境インデックス |
| `role` | `caregiver` / `recipient` | エージェントの役割 |
| `motion_id` | `3` | モーションライブラリ内のID |
| `timestamp` | `45123` | ミリ秒下5桁（ファイル名衝突回避） |
| `data_type` | `torques` | 5種類のいずれか |

### 各 `.npy` ファイルの Shape

| ファイル名末尾 | Shape | dtype |
|---|---|---|
| `_torques.npy` | `[T, num_dofs]` | float32/64 |
| `_joint_angles.npy` | `[T, num_dofs]` | float32/64 |
| `_root_positions.npy` | `[T, 3]` | float32/64 |
| `_root_orientations_quat.npy` | `[T, 4]` | float32/64 |
| `_root_orientations_6d.npy` | `[T, 6]` | float32/64 |

`T` = エピソード長（ステップ数）、最低6ステップ以上のエピソードのみ保存される。

---

## 保存データの読み込み例

```python
import numpy as np

torques = np.load("output/HumanoidIm/my_exp/rollout_data/env_0_..._torques.npy")
joint_angles = np.load("output/HumanoidIm/my_exp/rollout_data/env_0_..._joint_angles.npy")
root_pos = np.load("output/HumanoidIm/my_exp/rollout_data/env_0_..._root_positions.npy")
root_quat = np.load("output/HumanoidIm/my_exp/rollout_data/env_0_..._root_orientations_quat.npy")
root_6d = np.load("output/HumanoidIm/my_exp/rollout_data/env_0_..._root_orientations_6d.npy")

print(f"Episode length: {torques.shape[0]} steps")
print(f"Number of joints: {torques.shape[1]}")
```

---

## 他のコードベースへの移植チェックリスト

1. **`__init__`** に `self.recipient_torque_buffer = []` を追加
2. **`post_physics_step`** に `_track_recipient_torques()` の呼び出しを追加（条件付き）
3. **`_track_recipient_torques`** メソッドを実装
   - 取得元テンソル名（`dof_force_tensor`, `_dof_pos`, `_rigid_body_pos`, `_rigid_body_rot`）を自分の環境のテンソル名に合わせる
   - ロール判定ロジックを自分のタスクに合わせる
   - `motion_lib` 関連は不要なら省略可
4. **`_reset_envs`** に `_save_torque_data_for_reset_envs()` の呼び出しを追加（`super()` の **前**）
5. **`_save_torque_data_for_reset_envs`** メソッドを実装
6. **`_get_motion_filename_from_data`** ヘルパーを実装（またはシンプルな固定文字列に置き換え）
7. **`flags`** の仕組みがない場合は、独自のフラグや config で保存の ON/OFF を制御する
8. `quat_to_tan_norm` が不要な場合（6D回転表現が不要な場合）は省略可

### 最小構成（motion_lib やロール不要の場合）

```python
# __init__
self.rollout_buffer = []

# post_physics_step
def post_physics_step(self):
    super().post_physics_step()
    if self.save_rollout_data:  # configで制御
        for env_id in range(self.num_envs):
            self.rollout_buffer.append({
                'env_id': env_id,
                'step': self.progress_buf[env_id].item(),
                'torques': self.dof_force_tensor[env_id].cpu().numpy().copy(),
                'joint_angles': self._dof_pos[env_id].cpu().numpy().copy(),
                'root_position': self._rigid_body_pos[env_id, 0, :3].cpu().numpy().copy(),
            })

# _reset_envs
def _reset_envs(self, env_ids):
    if self.save_rollout_data:
        self._save_rollout_data(env_ids)
    super()._reset_envs(env_ids)

# 保存
def _save_rollout_data(self, env_ids):
    import os, time
    for env_id in env_ids:
        env_data = [d for d in self.rollout_buffer if d['env_id'] == env_id.item()]
        if len(env_data) <= 5:
            self.rollout_buffer = [d for d in self.rollout_buffer if d['env_id'] != env_id.item()]
            continue
        output_dir = f"output/HumanoidIm/{self.cfg.exp_name}/rollout_data"
        os.makedirs(output_dir, exist_ok=True)
        ts = int(time.time() * 1000) % 100000
        base = f"env{env_id.item()}_{ts}"
        np.save(os.path.join(output_dir, f"{base}_torques.npy"), np.array([d['torques'] for d in env_data]))
        np.save(os.path.join(output_dir, f"{base}_joint_angles.npy"), np.array([d['joint_angles'] for d in env_data]))
        np.save(os.path.join(output_dir, f"{base}_root_positions.npy"), np.array([d['root_position'] for d in env_data]))
        self.rollout_buffer = [d for d in self.rollout_buffer if d['env_id'] != env_id.item()]
```

---

## 注意事項

- **メモリ使用量**: バッファはPythonリスト（CPU上のNumPy配列）なので、環境数 × エピソード長が大きいとメモリを消費する。長時間のテストでは定期的にバッファをクリアするか、保存を限定的にすること。
- **`_reset_envs` の呼び出し順**: `_save_torque_data_for_reset_envs` は必ず `super()._reset_envs()` の**前**に呼ぶ。リセット後はテンソルが上書きされるため。
- **現在のコメントアウト状態**: `humanoid_im_hhi_assist_bed.py` の `_reset_envs` 内の `_save_torque_data_for_reset_envs` 呼び出しは現在コメントアウトされている（L3346）。有効化するにはコメントを解除する必要がある。
- **`flags` の設定**: `flags.test` と `flags.im_eval` は、テスト実行時のコマンドライン引数で設定される。通常のトレーニング中はデータ収集は行われない。
