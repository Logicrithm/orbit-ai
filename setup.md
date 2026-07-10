1. AMD Developer Cloud → GPU Droplet → MI300X x1 → vLLM Quick Start image
2. ssh root@<ip>
3. docker exec -it rocm /bin/bash
4. export HF_TOKEN=hf_xxx
5. nohup vllm serve google/gemma-3-4b-it --host 0.0.0.0 --port 8000 --max-model-len 8192 > /tmp/vllm.log 2>&1 &
6. tail -f /tmp/vllm.log   → wait for "Application startup complete"  (~2 min)
7. curl http://localhost:8000/v1/models