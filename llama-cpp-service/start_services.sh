#!/bin/bash

# Check if MODEL_PATH is set (instead of MODEL_NAME for llama.cpp)
if [ -z "$MODEL_NAME" ]; then
    echo "MODEL_PATH is not defined in .env file."
    exit 1
fi

# Start llama-cpp-python server
echo "Starting llama-cpp-python server in background..."
nohup python3 -m llama_cpp.server \
    --model ./models/"$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 11434 \
    --n_gpu_layers "$GPU_LAYERS" \
    --n_ctx "$N_CTX" \
    --n_batch "$N_BATCH" \
    > llamacpp.log 2>&1 &

# Save PID
echo $! > llamacpp.pid
echo "llama-cpp-python server started (PID: $(cat llamacpp.pid))"

# Wait for server to initialize
echo "Waiting 5 seconds for server to initialize..."
sleep 5

# Test endpoint
echo "Checking server health..."
curl -s http://localhost:11434/v1/models > /dev/null
if [ $? -eq 0 ]; then
    echo "llama-cpp-python server is running and responding!"
else
    echo "Failed to connect to llama-cpp-python server."
    exit 1
fi
