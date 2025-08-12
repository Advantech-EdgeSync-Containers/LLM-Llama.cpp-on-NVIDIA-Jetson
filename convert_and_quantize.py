import os
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download


# ==== CONFIGURATION ====
repo_id = "<Hugging face Repo Id>"
llama_cpp_dir = "<Llama CPP dir>"
model_dir = "<Model directory>"
output_dir = "<Output directory to store model gguf and quantized models>"
model_name = "<Model Name>"
quant_type = "<Quantization Type>"
HF_TOKEN = "<Hugging face Token>"
# ==========================

def run(cmd):
    print(">>> Running:", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)
    print("Done\n")


def download_model():
    os.environ["HF_TOKEN"] = HF_TOKEN
    model_path = (f"{model_dir}/{model_name}")
    if not os.path.exists(model_path):
        snapshot_download(
            repo_id=repo_id,
            local_dir=f"models/{model_name}",
            use_auth_token=True
        )
    else:
        print("Model Path already exists")
        
def convert_to_gguf():
    convert_script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    output_path = Path(output_dir) / f"{model_name}.f16.gguf"
    cmd = [
        "python3", str(convert_script),
        str(model_dir),
        "--outfile", str(output_path),
        "--outtype", "f16"
    ]
    run(cmd)
    return output_path

def quantize_model(f16_path):
    quantize_bin = Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"
    quant_output = f16_path.with_name(f"{f16_path.stem}.{quant_type.lower()}.gguf")
    cmd = [
        str(quantize_bin),
        str(f16_path),
        str(quant_output),
        quant_type
    ]
    run(cmd)
    return quant_output

if __name__ == "__main__":
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Downloading HuggingFace model...")
    download_model()

    print("Converting HuggingFace model to GGUF...")
    f16_path = convert_to_gguf()

    print("Quantizing to", quant_type, "...")
    quant_path = quantize_model(f16_path)

    print("Full GGUF path:", f16_path)
    print("Quantized path:", quant_path)
