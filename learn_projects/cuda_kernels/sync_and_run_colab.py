import sys
import subprocess
import base64

SSH_HOST = "6a5S9SZSxGLwrvjnvu93c9tGY@sfo2.tmate.io"

def run_ssh(cmd):
    full_cmd = f"ssh -o StrictHostKeyChecking=no {SSH_HOST} \"{cmd}\""
    res = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    return res.stdout, res.stderr, res.returncode

def upload_file(local_path, remote_path):
    with open(local_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    
    cmd = f"echo '{encoded}' | base64 -d > {remote_path}"
    out, err, code = run_ssh(cmd)
    if code != 0:
        print(f"Error uploading {local_path}: {err}")
    else:
        print(f"Uploaded {local_path} -> {remote_path}")

if __name__ == "__main__":
    out, err, code = run_ssh("nvidia-smi")
    print("GPU Check:")
    print(out)
    
    run_ssh("mkdir -p /content/cuda_kernels")
    upload_file("/Users/puneeth/repo/ai_ms_python/learn_projects/cuda_kernels/00_gpu_fundamentals.py", "/content/cuda_kernels/00_gpu_fundamentals.py")
    upload_file("/Users/puneeth/repo/ai_ms_python/learn_projects/cuda_kernels/01_naive_matmul.py", "/content/cuda_kernels/01_naive_matmul.py")
    
    print("\nRunning Milestone 0 (GPU Fundamentals)...")
    out, err, code = run_ssh("cd /content/cuda_kernels && python3 00_gpu_fundamentals.py")
    print(out)
    if err:
        print("STDERR:", err)
