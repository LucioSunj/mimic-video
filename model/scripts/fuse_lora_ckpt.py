import argparse

import torch

ALPHA = 32


def fuse_ckpt(ckpt_path: str) -> str:
    ckpt = torch.load(ckpt_path)

    lora_rank = None

    for key in list(ckpt.keys()):
        if "lora_A" not in key:
            continue

        b_key = key.replace("lora_A", "lora_B")
        base_key = key.replace("lora_A.default", "base_layer")

        this_rank = ckpt[key].shape[0]
        lora_rank = lora_rank or this_rank
        assert lora_rank == this_rank, (lora_rank, this_rank)

        adapter = ckpt[b_key] @ ckpt[key]
        fused = ckpt[base_key] + ALPHA / this_rank * adapter

        del ckpt[key], ckpt[b_key], ckpt[base_key]

        ckpt[base_key.replace(".base_layer", "")] = fused

    print(f"{lora_rank=}")

    fused_path = ckpt_path.replace(".pt", "_fused.pt")
    torch.save(ckpt, fused_path)

    return fused_path


def main():
    parser = argparse.ArgumentParser(description="Fuse adapters into weight.")
    parser.add_argument("ckpt_path", type=str, help="checkpoint path")
    args = parser.parse_args()

    fuse_ckpt(args.ckpt_path)


if __name__ == "__main__":
    main()
