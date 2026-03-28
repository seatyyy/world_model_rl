# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

# 使用 spawn 方式创建子进程，避免 fork 模式下 CUDA 上下文被错误继承
mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="opensora_libero_spatial_grpo_openvlaoft"
)
def main(cfg) -> None:
    # 校验配置合法性（检查 model_type、loss_type、adv_type 等是否匹配）
    cfg = validate_cfg(cfg)
    # 打印解析后的完整配置，所有 Hydra 变量插值已展开，便于调试
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    # 连接 Ray 集群（单机模式下 Ray 会自动启动；多机需提前手动 ray start）
    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    # 根据 cluster.component_placement 决定各组件（actor/rollout/env）放在哪些节点和 GPU 上
    component_placement = HybridComponentPlacement(cfg, cluster)

    # ---- Actor Worker：负责加载 VLA 模型、计算 loss、梯度更新 ----
    actor_placement = component_placement.get_strategy("actor")

    # 根据 loss_type 选择不同的 actor worker 实现
    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_dagger":
        from rlinf.workers.actor.fsdp_dagger_policy_worker import (
            EmbodiedDAGGERFSDPPolicy,
        )

        actor_worker_cls = EmbodiedDAGGERFSDPPolicy
    else:
        # PPO / GRPO 等标准策略优化算法走此分支
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

        actor_worker_cls = EmbodiedFSDPActor
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # ---- Rollout Worker：加载 VLA 模型做推理，根据观测生成动作 ----
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # ---- Env Worker：管理环境交互（训练时可用世界模型如 OpenSora，评估时用真实仿真器如 LIBERO）----
    # env_type 从 cfg.env.train / cfg.env.eval 中读取，在 init_worker 时通过 get_env_cls() 动态加载
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    # 创建 EmbodiedRunner，驱动 env→rollout→actor 的训练循环
    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    # 初始化所有 worker（加载模型权重、创建优化器、初始化环境等）
    runner.init_workers()
    # 启动训练主循环：rollout 生成动作 → env 返回奖励 → actor 更新策略 → 同步权重 → 重复
    runner.run()


if __name__ == "__main__":
    main()
