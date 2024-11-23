"""Generic env stuff."""
import os
import subprocess


def envsetup(nogpu: bool = True, nojit: bool = False):
    """get gpu info, possibly disable gpu in notebook."""
    if nogpu:
        os.environ |= ({
            "JAX_PLATFORM_NAME": "cpu",
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "-1",
        })
    if nojit: 
        os.environ["JAX_DISABLE_JIT"] = "1"
    try:
        gpu_info = []
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0: raise Exception()
        output = result.stdout.strip()
        for line in output.split('\n'):
            name, total, used, free = line.split(',')
            gpu_info.append({
                "GPU Name": name.strip(),
                "Total Memory (MB)": int(total.strip()),
                "Used Memory (MB)": int(used.strip()),
                "Free Memory (MB)": int(free.strip())
            })
    except Exception as e:
        print(f"ERR getting GPU info: {e}")
    return gpu_info
