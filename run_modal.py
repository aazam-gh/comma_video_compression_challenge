import modal

app = modal.App("quantizr-compress")

# Build an image with all dependencies + CUDA
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("ffmpeg")
    .pip_install(
        "numpy",
        "einops",
        "timm",
        "safetensors",
        "segmentation-models-pytorch",
        "tqdm",
        "pillow",
        "av",
        "charset-normalizer",
        "requests",
        "urllib3",
        "brotli",
        "torch",
        "torchvision",
        "nvidia-dali-cuda120",
        extra_index_url="https://pypi.nvidia.com",
    )
)

# Mount local repo (read-only) so remote container can access source + videos
LOCAL_DIR = "/root/comma_video_compression_challenge"


@app.function(
    image=image,
    gpu="A10G",  # 24GB VRAM — plenty for batch_size=4
    timeout=7200,  # 2 hour max
    memory=32768,  # 32GB RAM
    mounts=[
        modal.Mount.from_local_dir(
            local_path="/Users/alcadeus/Github/comma_video_compression_challenge",
            remote_path=LOCAL_DIR,
            condition=lambda p: not any(
                p.startswith(s)
                for s in [".venv", ".git", "__pycache__", "archive", "node_modules"]
            ),
        )
    ],
)
def run_compress():
    import subprocess

    result = subprocess.run(
        [
            "python3",
            f"{LOCAL_DIR}/submissions/quantizr/compress.py",
            "--batch-size",
            "4",
            "--device",
            "cuda:0",
        ],
        cwd=LOCAL_DIR,
        capture_output=True,
        text=True,
    )
    print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr)
        raise RuntimeError("Pipeline failed")
    print("Pipeline complete!")


@app.local_entrypoint()
def main():
    run_compress.remote()
