# 迁移测试脚本计划

我将把所有的测试相关脚本统一整理到 `tests/` 文件夹下，保持项目结构整洁。

## 1. 目标文件
- **源路径**: `scripts/verify_update.py` (刚刚创建的验证脚本)
- **目标路径**: `tests/verify_update.py`

## 2. 实施步骤
1.  **移动文件**: 将 `scripts/verify_update.py` 移动到 `tests/` 目录。
2.  **验证引用**: 检查脚本中的 `sys.path` 路径添加逻辑。
    - 原逻辑: `sys.path.append(str(Path(__file__).resolve().parent.parent))`
    - `tests/` 目录与 `scripts/` 目录同级，都是项目根目录的子目录，因此该逻辑依然有效，能正确指向项目根目录。
3.  **验证运行**: 运行迁移后的脚本，确保功能正常。

## 3. 现有文件概览
目前 `tests/` 目录下已有：
- `test_multiturn_dialogue.py`

迁移后，所有测试脚本将集中管理。
