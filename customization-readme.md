# Model Customization & Environment Configuration

## Model Customization 
If needed, llama-cpp-python allows tuning several inference parameters at runtime to influence output quality, creativity, and behavior. These can be configured by passing via API payload.

| Parameter      | Description                |
|----------------|----------------------------|
| temperature    | Controls randomness. Lower = more deterministic; higher = more creative/unpredictable. Range: 0.0–2.0 |
| repeat_penalty | Penalizes repetition in output. Values greater than 1.0 reduce repeated phrases. Default is around 1.1 |
| top_p          | Enables nucleus sampling—limits token selection to a cumulative probability (e.g., 0.95). Helps control diversity |
| top_k          | Limits sampling to the top K most probable next tokens (e.g., 50). Lower values = more focused output |
| max_tokens     | Limits the number of output tokens in a single generation.  |
| stop          | List of stop strings. Generation halts if any stop sequence is encountered. |

For other parameters refer to this official documentation from [**LlamaCpp**](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/). The above parameters can be customized by passing them via API payload.

## Environment Configuration

The `.env` file allows you to customize the runtime behavior of the container using environment variables.

### Key Environment Variables
``` bash
# --- Model Settings ---
# Model Name to be used by LlamaCPP
MODEL_NAME=Meta-Llama-3.2-1B-Instruct.gguf

# --- Docker Compose File Settings---
COMPOSE_FILE_PATH=./docker-compose.yml

# --- Open Web UI Settings ---
OPENWEBUI_PORT=3000
OPENAI_API_LLAMA_CPP_BASE=http://localhost:11434/v1

# --- LlamaCpp Settings ---
GPU_LAYERS=30
N_CTX=2048
N_BATCH=512

```


## LlamaCpp External Access
- Add the HOST flag when starting the llamacpp server and point it to 0.0.0.0. This can be added on the Python interface, CLI flags, or environment variables.
  ```
    --host=0.0.0.0
  ```
- Now verify that LlamaCPP should be accessible on the following address on a browser externally
  ```
    http://<device_ip>:11434/docs
  ```
  It should display Swagger documentation. 

![ollama-status](data%2Fimages%2Fllamacpp-python-curl.png)