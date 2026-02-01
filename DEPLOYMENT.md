# Deployment Guide

## Quick Start

### Local Development

1. **Clone Repository**
```bash
git clone https://github.com/[your-username]/medassist-agentic-workflow.git
cd medassist-agentic-workflow
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Model (Optional)**
```bash
# If you want to use the actual MedGemma model
# Otherwise it will run in demo mode
python download_model.py
```

5. **Run Demo Application**
```bash
python app.py
```

6. **Access Interface**
Open browser to: http://localhost:7860

---

## Docker Deployment

### Build Image

```bash
docker build -t medassist:latest .
```

### Run Container

```bash
docker run -p 7860:7860 --gpus all medassist:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  medassist:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Cloud Deployment

### Hugging Face Spaces

1. Create new Space on Hugging Face
2. Select "Gradio" SDK
3. Upload all files
4. Add secrets for any API keys
5. Space will auto-deploy

**Space URL:** https://huggingface.co/spaces/[username]/medassist

### AWS EC2

**Recommended Instance:** g4dn.xlarge (1× NVIDIA T4 GPU)

```bash
# Launch EC2 instance
# SSH into instance
ssh -i key.pem ubuntu@[instance-ip]

# Install dependencies
sudo apt update
sudo apt install python3-pip nvidia-cuda-toolkit

# Clone and setup
git clone [repo-url]
cd medassist-agentic-workflow
pip install -r requirements.txt

# Run with nohup
nohup python app.py > app.log 2>&1 &
```

### Google Cloud Run (CPU-only)

```bash
# Build container
gcloud builds submit --tag gcr.io/[project-id]/medassist

# Deploy
gcloud run deploy medassist \
  --image gcr.io/[project-id]/medassist \
  --platform managed \
  --memory 8Gi \
  --cpu 4 \
  --timeout 300
```

---

## Production Considerations

### Performance Optimization

1. **Model Quantization**
```python
# Use 8-bit quantization
orchestrator = MedAssistOrchestrator(
    model_name="google/medgemma-2b",
    device="cuda",
    load_in_8bit=True  # Reduces VRAM by 50%
)
```

2. **Batch Processing**
```python
# Process multiple cases in parallel
results = orchestrator.batch_process(cases, max_parallel=4)
```

3. **Caching**
```python
# Cache common queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_guideline(condition):
    return ClinicalGuidelines.get_guideline(condition)
```

### Security

1. **HIPAA Compliance**
- All data processing local
- Audit logging enabled
- Encryption at rest and in transit
- User authentication required

2. **API Security**
```python
# Add authentication
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/process")
async def process_case(case: dict, token: str = Depends(security)):
    if not verify_token(token):
        raise HTTPException(status_code=401)
    return orchestrator.process_case(case)
```

3. **Rate Limiting**
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.route("/process")
@limiter.limit("10/minute")
def process_case():
    pass
```

### Monitoring

1. **Application Metrics**
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram

cases_processed = Counter('cases_processed_total', 'Total cases processed')
processing_time = Histogram('processing_seconds', 'Time to process case')

@processing_time.time()
def process_case(case):
    result = orchestrator.process_case(case)
    cases_processed.inc()
    return result
```

2. **Health Checks**
```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": orchestrator.model is not None,
        "gpu_available": torch.cuda.is_available()
    }
```

3. **Error Tracking**
```python
import sentry_sdk

sentry_sdk.init(dsn="your-dsn")
```

### Scaling

1. **Horizontal Scaling**
- Use load balancer (NGINX, HAProxy)
- Deploy multiple instances
- Share model via network storage

2. **Model Serving**
```python
# Use TorchServe for production
torchserve --start \
  --model-store models \
  --models medgemma=medgemma.mar
```

3. **Queue-based Processing**
```python
# Use Celery for async processing
from celery import Celery

app = Celery('medassist', broker='redis://localhost:6379')

@app.task
def process_case_async(case):
    return orchestrator.process_case(case)
```

---

## Configuration

### Environment Variables

```bash
# .env file
MODEL_NAME=google/medgemma-2b
MODEL_PATH=/models/medgemma-2b
DEVICE=cuda
LOAD_IN_8BIT=true
MAX_LENGTH=2048
TEMPERATURE=0.7

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/medassist.log

# Security
API_KEY=your-secret-key
ENABLE_AUTH=true

# Performance
MAX_PARALLEL_AGENTS=3
TIMEOUT_SECONDS=120
```

### Configuration File

```yaml
# config.yaml
model:
  name: google/medgemma-2b
  device: auto
  quantization: 8bit
  max_length: 2048

agents:
  max_iterations: 10
  timeout: 120
  parallel_execution: true

security:
  audit_logging: true
  deidentify_output: true
  
deployment:
  host: 0.0.0.0
  port: 7860
  workers: 4
```

---

## Maintenance

### Model Updates

```bash
# Update model quarterly
python scripts/update_model.py --version 2024-Q1

# Retrain with new data
python scripts/finetune.py \
  --base-model google/medgemma-2b \
  --data data/new_cases.json \
  --output models/medgemma-updated
```

### Database Backups

```bash
# Backup case history and audit logs
python scripts/backup.py --output backups/$(date +%Y%m%d)
```

### Performance Testing

```bash
# Load testing
python scripts/load_test.py --cases 1000 --concurrent 10

# Benchmark
python scripts/benchmark.py --iterations 100
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Solution: Enable quantization or use smaller batch size
export CUDA_VISIBLE_DEVICES=0
python app.py --load-in-8bit
```

**2. Slow Response Times**
```bash
# Check GPU utilization
nvidia-smi

# Profile code
python -m cProfile -o profile.out app.py
```

**3. Model Not Loading**
```bash
# Check model files
ls -lh models/

# Download manually
huggingface-cli download google/medgemma-2b
```

### Logs

```bash
# View application logs
tail -f /var/log/medassist.log

# View Docker logs
docker logs -f medassist

# Enable debug mode
export LOG_LEVEL=DEBUG
python app.py
```

---

## Support

- **Issues:** https://github.com/[your-repo]/issues
- **Documentation:** https://[your-docs-site]
- **Email:** [your-email]
- **Slack/Discord:** [community-link]
