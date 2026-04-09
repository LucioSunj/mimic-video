from __future__ import annotations

from imaginaire.utils import distributed
from imaginaire.utils.callback import Callback
from scripts.fuse_lora_ckpt import fuse_ckpt

# import subprocess as sp


class VideoEvalCallback(Callback):
    def __init__(self, fuse_lora: bool):
        self._fuse_lora = fuse_lora

    @distributed.rank0_only
    def on_save_checkpoint_success(
        self, iteration: int = 0, elapsed_time: float = 0, checkpoint_path: str | None = None
    ) -> None:
        if self._fuse_lora:
            checkpoint_path = fuse_ckpt(checkpoint_path)

        try:
            # sp.run(
            #     [submit job to cluster management that generates videos from snippets],
            #     timeout=5,
            # )
            pass
        except Exception:
            pass
