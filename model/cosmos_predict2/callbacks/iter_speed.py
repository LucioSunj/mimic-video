# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time

import torch
from torch import Tensor

from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import log
from imaginaire.utils.distributed import rank0_only


class IterSpeed(EveryN):
    """
    Args:
        hit_thres (int): Number of iterations to wait before logging.
    """

    def __init__(self, *args, hit_thres: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = None
        self.hit_counter = 0
        self.hit_thres = hit_thres
        self.name = self.__class__.__name__
        self.last_hit_time = time.time()
        self.scalar_log_path = None

    @staticmethod
    def _to_scalar(value):
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            return float(value.detach().float().cpu().item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _collect_scalars(self, output_batch: dict[str, Tensor], loss: Tensor, iteration: int) -> dict[str, float]:
        scalars = {"iteration": float(iteration), "loss": float(loss.detach().float().cpu().item())}
        for key, value in output_batch.items():
            scalar = self._to_scalar(value)
            if scalar is not None:
                scalars[key] = scalar

        x0_mse = scalars.get("source/source_vs_x0_mse")
        gaussian_mse = scalars.get("source/source_vs_gaussian_mse")
        if x0_mse is not None and gaussian_mse is not None:
            scalars["source/source_vs_gaussian_ratio"] = x0_mse / (gaussian_mse + 1e-8)
        return scalars

    def _write_scalar_log(self, scalars: dict[str, float]) -> None:
        if self.scalar_log_path is None:
            self.scalar_log_path = os.path.join(self.config.job.path_local, "train_scalars.jsonl")
            os.makedirs(os.path.dirname(self.scalar_log_path), exist_ok=True)
        with open(self.scalar_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(scalars, sort_keys=True) + "\n")

    @staticmethod
    def _format_scalar_summary(scalars: dict[str, float]) -> str:
        keys = [
            "probe/sampled_action_mse_gtvid",
            "loss/flow",
            "loss/source_prior",
            "loss/source_kl",
            "source/source_vs_x0_mse",
            "source/source_vs_x0_over_var",
            "source/source_vs_gaussian_mse",
            "source/source_vs_gaussian_ratio",
            "source/mu_vs_x0_mse",
            "source/source_vs_mu_mse",
            "source/std_mean",
            "source/std_min",
            "source/logstd_mean",
            "source/logstd_floor_frac",
            "source/logstd_ceiling_frac",
            "source/mu_std",
            "source/mu_batch_std",
            "source/mu_horizon_std",
            "Var_inst[x_0]",
        ]
        parts = []
        for key in keys:
            value = scalars.get(key)
            if value is not None:
                parts.append(f"{key}={value:.6g}")
        return " | ".join(parts)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.hit_counter < self.hit_thres:
            log.info(
                f"Iteration {iteration}: "
                f"Hit counter: {self.hit_counter + 1}/{self.hit_thres} | "
                f"Loss: {loss.item():.4f} | "
                f"Time: {time.time() - self.last_hit_time:.2f}s"
            )
            self.hit_counter += 1
            self.last_hit_time = time.time()
            #! useful for large scale training and avoid oom crash in the first two iterations!!!
            torch.cuda.synchronize()
            return
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    @rank0_only
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, Tensor],
        output_batch: dict[str, Tensor],
        loss: Tensor,
        iteration: int,
    ) -> None:
        if self.time is None:
            self.time = time.time()
            return
        cur_time = time.time()
        iter_speed = (cur_time - self.time) / self.every_n / self.step_size

        scalars = self._collect_scalars(output_batch, loss, iteration)
        self._write_scalar_log(scalars)
        scalar_summary = self._format_scalar_summary(scalars)
        suffix = f" | {scalar_summary}" if scalar_summary else ""
        log.info(f"{iteration} : iter_speed {iter_speed:.2f} seconds per iteration | Loss: {loss.item():.4f}{suffix}")

        self.time = cur_time
