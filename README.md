# LLM Llama.cpp on NVIDIA Jetson™

**Version:** 1.0
**Release Date:** November 2025
**Copyright:** © 2025 Advantech Corporation. All rights reserved.

## Overview
LLM Llama.cpp on NVIDIA Jetson™ offers a streamlined, hardware-accelerated platform for building and deploying conversational AI on NVIDIA Jetson™ devices. It features LlamaCpp-Python (a Python interface for LlamaCPP) and the Meta Llama 3.2 1B Instruct model, enabling efficient on-device inference. The container also integrates OpenWebUI for an intuitive chat interface and includes optimized AI software components. Designed for edge environments, it delivers high performance, low latency, and reliable real-time AI experiences.

## Key Features

| Feature                                  | Description                                                                   |
|------------------------------------------|-------------------------------------------------------------------------------|
| LlamaCpp Backend                     | Run large language models (LLMs) locally with simple setup and management     |
| Integrated OpenWebUI                 | Clean, user-friendly frontend for LLM chat interface                          |
| Meta Llama 3.2 1B Instruct Inference | Efficient on-device LLM via LlamaCpp-Python; minimal memory, high performance |
| REST API Access | OpenAI-compatible APIs for model interaction                                   |
| Flexible Parameters                  | Adjust inference with `n_ctx`, `n_threads`, `n_gpu_layers`, etc.              |
| Prompt Templates                     | Supports formats like `chatml`, `llama`, and more                             |
| Offline Capability                   | Fully offline after container image setup; no internet required               |



## Architecture
![llama-cpp-llama.png](data%2Farchitectures%2Fllama-cpp-llama.png)

## Repository Structure
```
LLM-Llama.cpp-on-NVIDIA-Jetson/
├── .env                                        # Environment configuration
├── build.sh                                    # Build helper script
├── wise-bench.sh                               # Wise Bench script
├── docker-compose.yml                          # Docker Compose setup
├── README.md                                   # Overview
├── quantization-readme.md                      # Model quantization steps
├── other-AI-capabilities-readme.md             # Other AI capabilities supported by container image
├── llm-models-performance-notes-readme.md      # Performance notes of LLM Models
├── efficient-prompting-for-compact-models.md   # Craft better prompts for small and quantized language models
├── customization-readme.md                     # Customization, optimization & configuration guide
├── .gitignore                                  # Git ignore specific files
├── convert_and_quantize.py                     # Script to convert Hugging face models to gguf 
├── download_model.sh                           # Hugging Face model downloader
├── data                                        # Contains subfolders for assets like images, gifs etc.
└── llama-cpp-service/                          # Folder containing LlamaCPP files
    ├── models                                  # Contains model files
    └── start_services.sh                       # Startup script
```
## Container Description

### Quick Information

`build.sh` will start following two containers:

| Container Name | Description                                                                                                                                       |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| LLM-Llama.cpp-on-NVIDIA-Jetson | Provides a hardware-accelerated development environment using various AI software components along with Meta Llama 3.2 1B Instruct & LlamaCpp-Python |
| openweb-ui-service | Optional, provides UI which is accessible via browser for inferencing                                                                             |

### LLM Llama.cpp on NVIDIA Jetson™ Container Highlights

This container leverages [**LlamaCpp**] as the local inference engine to serve LLMs efficiently on NVIDIA Jetson™ devices. LlamaCpp provides a lightweight and container-friendly API layer for running language models without requiring cloud-based services.

| Feature                        | Description                                                                    |
|-------------------------------|--------------------------------------------------------------------------------|
| Local Inference Engine    | On-device model (Meta Llama 3.2 1B Instruct) inference via LlamaCpp’s REST API. |
| Python Bindings           | Simplified integration using LlamaCpp-Python for developers building AI pipelines in Python            |
| OpenAI API Compatibility | Supports OpenAI Chat Completions API; works with LangChain, OpenWebUI. |
| Streaming Output Support  | Real-time token streaming for chat UIs.                                                                |
| Edge Optimized            | Works with quantized `.gguf` models                                                                    |
| Model Management          | Pull from Hugging Face or use your own .gguf models                                                                         |
| Prompt Engineering        | Supports system/user prompt separation and composition.                                                |
| Offline-First             | No internet needed after model download; ensures privacy.                                              |
| Developer Friendly        | Simple CLI and Docker support for local deployment.                                                    |
| Easy Integration          | Backend-ready for LangChain, FastAPI, RAG, UIs, etc.                                                   |
| AI Dev Environment        | Full HW-accelerated container for AI development.                                                      |


### OpenWebUI Container Highlights

OpenWebUI serves as a clean and responsive frontend interface for interacting with LLMs via LlamaCpp-Python's OpenAI-compatible endpoints. When containerized, it provides a modular, portable, and easily deployable chat interface suitable for local or edge deployments.

| Feature                          | Description |
|----------------------------------|-------------|
| User-Friendly Interface      | Sleek, chat-style UI for real-time interaction. |
| OpenAI-Compatible Backend    | Works with LlamaCpp, OpenAI, and similar APIs with minimal setup. |
| Container-Ready Design       | Lightweight and optimized for edge or cloud deployments. |
| Streaming Support            | Enables real-time response streaming for interactive UX. |
| Authentication & Access Control | Basic user management for secure access. |
| Offline Operation            | Runs fully offline with local backends like LlamaCpp. |


## List of READMEs

| Module   | Link                | Description                     |
|----------|----------------------------|---------------------------------|
| Quick Start | [README](./README.md) | Overview of the container image   |
| Customization & optimization | [README](./customization-readme.md) | Steps to customize a model, configure environment, and optimize |
| Model Performances | [README](./llm-models-performance-notes-readme.md) | Performance stats of various LLM Models  |
| Other AI Capabilities  | [README](./other-AI-capabilities-readme.md) | Other AI capabilities supported by the container |
| Quantization  | [README](./quantization-readme.md) | Steps to quantize a model |
| Prompt Guidelines   | [README](./efficient-prompting-for-compact-models.md) | Guidelines to craft better prompts for small and quantized language models |

## Model Information  

This image uses Meta Llama 3.2 1B. For inferencing, here are the details about the model used:

| Item  | Description                  |
|---------|------------------------------|
| Model source | Hugging Face - Meta-Llama-3.2-1B-Instruct |
| Model architecture  | llama                        |
| Model quantization | Q4_K_M                         |
| Number of Parameters | ~1.24 B |
| Model size  | ~0.8 GB                      |
| Default context size  | 2048                         |

## Hardware Specifications

| Component | Specification |
|-----------|---------------|
| Target Hardware | NVIDIA Jetson™ |
| GPU | NVIDIA® Ampere architecture with 1024 CUDA® cores |
| DLA Cores | 1 (Deep Learning Accelerator) |
| Memory | 4/8/16 GB shared GPU/CPU memory |
| JetPack Version | 6.0 |

## Software Components

The following software components are available in the base image:

| Component | Version   | Description                        |
|-----------|-----------|------------------------------------|
| CUDA®     | 12.6.68   | GPU computing platform             |
| cuDNN     | 9.3.0.75  | Deep Neural Network library        |
| TensorRT™ | 10.3.0.30 | Inference optimizer and runtime    |
| VPI       | 3.2.4     | Vision Programming Interface       |
| Vulkan    | 1.3.204   | Graphics and compute API           |
| OpenCV    | 4.8.0     | Computer vision library with CUDA® |


The following software components/packages are provided further as a part of this image:

| Component                  | Version | Description |
|----------------------------|---------|-------------|
| LlamaCpp         | 0.2.0   | LLM inference engine                                             |
| LlamaCpp-Python  | 0.3.9   | Python wrapper for LlamaCPP                                      |
| OpenWebUI                  | 0.6.5   | Provided via separate OpenWebUI container for UI  |
| Meta Llama 3.2 1B Instruct | N/A     | Pulled/Stored inside LlamaCpp container and persisted  |


## Before You Start
Please take a note of the following points:

- The container provides flexibility to users, as they can download the pre-converted & pre-quantized Meta Llama 3.2 Instruct 1B model from Hugging Face using `download_model.sh`, or they can also follow [the Quantization README](./quantization-readme.md) to convert & quantize Hugging Face models by themselves.

- In case users convert & quantize their own models, please ensure that the models are placed under the `/models` directory and `MODEL_NAME` is also updated in the `.env` file before starting the services.

## Quick Start

### Installation

```
# Clone the repository
git clone https://github.com/Advantech-EdgeSync-Containers/LLM-Llama.cpp-on-NVIDIA-Jetson.git
cd LLM-Llama.cpp-on-NVIDIA-Jetson

# Update HF_TOKEN in .env file
# Create a hugging face token with read permissions
# Follow 'Authentication token' section under quantization-readme.md
HF_TOKEN=<ADD-YOUR-HF-TOKEN>

# Make the download model script executable
chmod +x download_model.sh

# Download the hugging face model
sudo ./download_model.sh

# Make the build script executable
chmod +x build.sh

# Launch the container
sudo ./build.sh
```

### Run Services

After installation succeeds, by default control lands inside the container. Run the following command to start services within the container.

```
# Under /workspace, run this command
# Provide executable rights
chmod +x start_services.sh

# Start services
./start_services.sh
```
Allow some time for the OpenWebUI and Jetson™ LLM LlamaCpp container to settle and become healthy.

### AI Accelerator and Software Stack Verification (Optional)
```
# Verify AI Accelerator and Software Stack Inside Docker Container
chmod +x /workspace/wise-bench.sh
./workspace/wise-bench.sh
```

![llama-cpp-wise-bench.png](data%2Fimages%2Fllama-cpp-wise-bench.png)

Wise-bench logs are saved in `wise-bench.log` file under `/workspace`

### Check Installation Status
Exit from the container and run the following command to check the status of the containers:
```
sudo docker ps
```
Allow some time for containers to become healthy.

### UI Access
Access OpenWebUI via any browser using the URL given below. Create an account and perform a login:
```
http://localhost_or_Jetson_IP:3000
```

### Quick Demonstration:

![Demo](data%2Fgifs%2Fllamacpp-metallama.gif)

## Prompt Guidelines

This [README](./efficient-prompting-for-compact-models.md) provides essential prompt guidelines to help you get accurate and reliable outputs from small and quantized language models.

## LlamaCpp Logs and Troubleshooting

### Log Files

Once services have been started inside the container, the following log files are generated:

| Log File | Description |
|-----------|---------|
| llamacpp.pid | Provides process-id for the currently running LlamaCpp service   |
| llamacpp.log | Provides LlamaCpp service logs |

### Troubleshoot

Here are quick commands/instructions to troubleshoot issues with the Jetson™ LLM LlamaCpp Container:

- View LlamaCpp service logs within the LlamaCpp container
  ```
  tail -f llamacpp.log
  ```
- Check if the model is loaded using CPU or GPU or partially both via logs (ideally should be 100% GPU loaded).

- Kill & restart services within the container (check pid manually via `ps -eaf` or use pid stored in `LlamaCpp.pid`)
  ```
  kill $(cat llamacpp.pid)
  ./start_services.sh
  ```

  Confirm there is no LlamaCpp service running using:
  ```
  ps -eaf
  ```

## LlamaCpp Python Inference Sample
Here's a Python example to draw inference using the LlamaCpp-Python API. This script sends a prompt to an LlamaCpp model (e.g., Meta Llama3.2 1B, DeepSeek R1 1.5B, etc.) and retrieves the response.


### Run script
Run either of the following scripts inside the container using Python.

#### Option 1: Use the llama-cpp-python server running locally
```
import requests
import json

def generate_with_LlamaCpp_stream(prompt):
    url = "http://localhost:11434/v1/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": prompt
    }

    with requests.post(url, json=payload, headers=headers, stream=True) as response:
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        print("Generated Response:\n")
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    print(token, end="", flush=True)
                except json.JSONDecodeError as e:
                    print(f"[Error decoding JSON chunk]: {e}")

# Example usage
if name == "__main__":
    prompt = "Explain quantum computing in simple terms."
    generate_with_LlamaCpp_stream(prompt)

```
#### Option 2: Direct Python API (llama-cpp-python). This avoids HTTP calls entirely and uses the Llama class directly.
```
from llama_cpp import Llama

llm = Llama(model_path="./models/Meta-Llama-3.2-1B-Instruct.gguf", n_ctx=2048)

def generate_response(prompt):
    response = llm(prompt, stream=True)
    print("Generated Response:\n")
    for chunk in response:
        print(chunk["choices"][0]["text"], end="", flush=True)

if name == "__main__":
    generate_response("Explain quantum computing in simple terms.")
```
```
Save it as script.py and run it using the following command:
```
python3 script.py
```
The inference stream should get started after running this.

## Best Practices and Recommendations

### Memory Management & Speed
- Ensure models are fully loaded into GPU memory for best results.
- Use quantized GGUF models for the best performance & accuracy.
- Batch inference for better throughput
- Use stream processing for continuous data
- Enable Jetson™ Clocks for better inference speed
- Increase swap size if models loaded are large
- Use lesser context & batch size to avoid high memory utilization
- Set max-tokens in API payloads to avoid unnecessarily long response generations, which may affect memory utilization.
- It is recommended to use models with parameters <2B and Q4* quantization.

### LlamaCpp Model Behavior Corrections 
- Restart LlamaCpp services
- Check if the model is correctly loaded into the GPU or not via logs
- Tweak model parameters as needed

## REST API Access

[**Official Documentation**](https://github.com/abetlen/llama-cpp-python)

### LlamaCpp Python APIs Swagger
LlamaCpp APIs are accessible on the default endpoint (unless modified). 

![llama-cpp-python-curl.png](data%2Fimages%2Fllamacpp-python-curl.png)

For further details, please refer to the official documentation of LlamaCpp Python as mentioned on top.

## Known Limitations

1. RAM Utilization: RAM utilization for running this container image occupies approximately 6 GB RAM when running on NVIDIA® Orin™ NX – 8 GB. Running this image on Jetson™ Nano may require some additional steps, like increasing swap size or using lower quantization as suited. 
2. OpenWebUI Dependencies: When OpenWebUI is started for the first time, it installs a few dependencies that are then persisted in the associated Docker volume. Allow it some time to set up these dependencies. This is a one-time activity. 
3. DeepSeek R1 1.5B Model Memory Utilization: DeepSeek R1 1.5B (in GGUF) format has been found to have higher memory utilization, which may increase with prolonged inferences. It is recommended to use lower context/batch size and also pass max-tokens via API requests to keep this behavior in check. This issue is not applicable for models like Meta Llama 3.2 1B, etc.


## Possible Use Cases

Leverage the container image to build interesting use cases like:

- Private LLM Inference on Local Devices: Run large language models locally with no internet requirement—ideal for privacy-critical environments

- Lightweight Backend for LLM APIs: Use LlamaCpp to expose models via its local API for fast integration with tools like LangChain, FastAPI, or custom UIs.

- Document-Based Q&A Systems: Combine LlamaCpp with a vector database to create offline RAG (Retrieval-Augmented Generation) systems for querying internal documents or manuals.

- Multilingual Assistants: Deploy multilingual chatbots using local models that can translate, summarize, or interact in different languages without depending on cloud services.

- LLM Evaluation and Benchmarking Easily swap and test different quantized models (e.g., Mistral, LLaMA, DeepSeek) to compare performance, output quality, and memory usage across devices.

- Custom Offline Agents: Use LlamaCpp as the reasoning core of intelligent agents that interact with other local tools (e.g., databases, APIs, sensors)—especially powerful when paired with LangChain

- Edge AI for Industrial Use: Deploy LlamaCpp on Edge to enable intelligent interfaces, command parsing, or decision-support tools at the edge.


Copyright © 2025 Advantech Corporation. All rights reserved.
