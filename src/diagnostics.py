"""
MicroInfer - Sub-Phase 0.1: System & GPU Hardware Diagnostics
Performs automated diagnostics on Python, PyTorch, CUDA, GPU properties, and dependencies.
"""

import sys
import json
import platform
import torch
from pathlib import Path


def run_diagnostics():
    """
    Executes full system diagnostics and returns a structured health report.
    """
    report = {
        "sub_phase": "0.1 - Environment & GPU Hardware Diagnostics",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }

    if torch.cuda.is_available():
        device_id = 0
        props = torch.cuda.get_device_properties(device_id)
        report["gpu_info"] = {
            "name": props.name,
            "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
            "total_vram_mb": round(props.total_memory / (1024 ** 2), 2),
            "multi_processor_count": props.multi_processor_count,
            "compute_capability": f"{props.major}.{props.minor}",
        }
    else:
        report["gpu_info"] = None

    # Verify installed library dependencies
    dependencies = {}
    for lib in ["transformers", "accelerate", "pytest", "numpy", "pandas"]:
        try:
            mod = __import__(lib)
            dependencies[lib] = getattr(mod, "__version__", "Installed")
        except ImportError:
            dependencies[lib] = "Missing"

    report["dependencies"] = dependencies
    return report


def main():
    print("=" * 60)
    print("  MICROINFER SUB-PHASE 0.1: HARDWARE & ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)

    report = run_diagnostics()

    print(f"\n[System Info]")
    print(f"  Python Version:   {report['python_version']}")
    print(f"  Platform:         {report['platform']}")
    print(f"  PyTorch Version:  {report['pytorch_version']}")
    print(f"  CUDA Available:   {report['cuda_available']}")

    if report['cuda_available']:
        print(f"  CUDA Version:     {report['cuda_version']}")
        print(f"  cuDNN Version:    {report['cudnn_version']}")
        gpu = report['gpu_info']
        print(f"\n[GPU Hardware Spec]")
        print(f"  Device Name:      {gpu['name']}")
        print(f"  Total VRAM:       {gpu['total_vram_gb']} GB ({gpu['total_vram_mb']} MB)")
        print(f"  SM Count:         {gpu['multi_processor_count']}")
        print(f"  Compute Arch:     SM {gpu['compute_capability']} (Ada Lovelace)")
    else:
        print("\n❌ WARNING: CUDA is NOT available! GPU acceleration disabled.")

    print(f"\n[Dependencies Check]")
    for lib, status in report['dependencies'].items():
        icon = "[OK]     " if status != "Missing" else "[MISSING]"
        print(f"  {icon} {lib:<15} {status}")

    # Export report to analysis/gpu_diagnostics.json
    output_dir = Path(__file__).parent.parent / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    out_path = output_dir / "gpu_diagnostics.json"

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[MicroInfer] Diagnostic report saved to '{out_path}'.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
