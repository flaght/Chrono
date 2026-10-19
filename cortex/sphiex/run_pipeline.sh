#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_pipeline.sh
# 作用: 在 uv-sirius 与 conda lingua 两个环境之间按顺序切换并执行指定的 Python 脚本
# ==============================================================================

# 遇错立即终止脚本（确保前序步骤失败时不会继续执行后续步骤）
set -eo pipefail

# 统一的工作目录
SPHIEX_DIR="/workspace/worker/pj/Chrono/cortex/sphiex"

# -----------------------------------------------------------------------------
# 1. uv-sirius 环境执行函数 (在子 Shell 中运行，执行完自动释放环境)
# -----------------------------------------------------------------------------
run_uv() {
    echo -e "\n\033[1;34m[Step: dichaos-uv]\033[0m 正在执行: $*"
    (
        cd "${SPHIEX_DIR}"
        source /workspace/worker/env/dichaos-uv/bin/activate
        export PYTHONPATH=.
        export PYTHONPATH="${SPHIEX_DIR}:/workspace/worker/pj/DiChaos${PYTHONPATH:+:${PYTHONPATH}}"
        # 执行传入的具体命令
        "$@"
    )
}

# -----------------------------------------------------------------------------
# 2. conda lingua 环境执行函数 (在子 Shell 中运行，执行完自动释放环境)
# -----------------------------------------------------------------------------
run_lingua() {
    echo -e "\n\033[1;32m[Step: conda-lingua]\033[0m 正在执行: $*"
    (
        cd "${SPHIEX_DIR}"
        source /opt/conda/bin/activate lingua
        export PYTHONPATH=.
        
        # 执行传入的具体命令
        "$@"
    )
}

# ==============================================================================
# 3. 顺序执行流程 (您可以在下方自由调整顺序、添加或替换执行脚本)
# ==============================================================================

echo "================ 开始执行流水线 ================"
#run_lingua python z10_live/1.0.1.create_features.py --method basic 
#run_uv python z10_live/1.0.1.create_features.py --method aggregation 
#run_lingua python z10_live/1.0.2.process_data.py
run_uv python z10_live/1.0.3.empirical_prediction.py

# run_lingua python z02_processing/1.0.4.process_data.py
# run_uv python z01_features/1.0.2.create_event_feature.py
# run_lingua python z01_features/1.0.0.create_returns.py
# run_lingua python z10_live/1.0.1.create_features.py --method basic 
# run_uv python z10_live/1.0.1.create_features.py --method aggregation 
# run_uv python z10_live/1.0.2.process_data.py

# run_uv python z10_live/1.0.3.empirical_prediction.py
# run_uv python z03_agents/2.0.8.empirical_prediction.py 

# 示例 步骤 1: 在 uv-sirius 环境下执行
#run_uv python -c "import sys; print('当前环境:', sys.executable)"
# run_uv python your_uv_step1.py

# 示例 步骤 2: 切换到 conda lingua 环境下执行
#run_lingua python z10_live/1.0.1.create_freatures.py

# 示例 步骤 3: 再次切回 uv-sirius 环境执行后续代码
# run_uv python your_uv_step2.py

# 示例 步骤 4: 切换到 conda lingua 环境执行评估或后处理
# run_lingua python z05_evaluate/evaluate.py

echo -e "\n\033[1;32m================ 所有步骤执行完成 ================\033[0m"