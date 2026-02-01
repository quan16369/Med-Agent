# Production Best Practices

## Overview
This document outlines production-ready best practices implemented in MedAssist.

## Table of Contents
1. [Configuration Management](#configuration-management)
2. [Logging & Monitoring](#logging--monitoring)
3. [Error Handling](#error-handling)
4. [Testing](#testing)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Deployment](#deployment)
7. [Security](#security)
8. [Performance](#performance)
9. [Observability](#observability)

## Configuration Management

### Environment-Based Configuration
```python
from config_production import get_config

# Automatically loads config based on environment
config = get_config()  # Uses MEDASSIST_ENV environment variable

# Or explicitly specify environment
dev_config = get_config("development")
prod_config = get_config("production")
```

### Configuration Hierarchy
1. **Environment Variables** (highest priority)
2. **Configuration File** (JSON)
3. **Default Values** (fallback)

### Environment Variables
```bash
# Core settings
export MEDASSIST_ENV=production
export CLINIC_ID=clinic_001
export REGION=southeast_asia

# Database
export DB_PATH=/var/lib/medassist/knowledge.db
export DB_CACHE_SIZE=20000

# Model
export MODEL_NAME=google/medgemma-2b
export MODEL_DEVICE=auto
export MODEL_4BIT=true

# Cloud
export CLOUD_API_KEY=your_api_key_here
export ENABLE_TELEMETRY=true

# Security
export ENABLE_ENCRYPTION=true
export ENABLE_AUTH=true

# Monitoring
export LOG_LEVEL=INFO
export LOG_FORMAT=json
```

### Configuration Validation
All configurations are validated on startup:
```python
config = get_config()
if not config.validate():
    logger.error("Configuration validation failed")
    sys.exit(1)
```

## Logging & Monitoring

### Structured Logging
All logs use JSON format in production:
```python
from medassist.logging_setup import setup_logging

setup_logging(
    log_level="INFO",
    log_format="json",
    log_file="/var/log/medassist/medassist.log",
    max_bytes=100*1024*1024,  # 100MB
    backup_count=10
)
```

### Log Levels
- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages (degraded operation)
- **ERROR**: Error messages (operation failed)
- **CRITICAL**: Critical errors (system failure)

### Audit Logging
Security-sensitive operations are logged separately:
```python
from medassist.logging_setup import AuditLogger

audit = AuditLogger()
audit.log_access(user_id, action, resource, success)
audit.log_auth_event(user_id, event, success, ip_address)
audit.log_case_access(user_id, case_id, action)
```

### Health Checks
Implements Kubernetes-style health probes:

#### Liveness Probe
```python
from medassist.health_checks import HealthChecker

checker = HealthChecker()
status = checker.check_liveness()
# Returns: healthy/unhealthy
```

#### Readiness Probe
```python
status = checker.check_readiness()
# Checks: database, model, disk, memory, custom components
```

#### Startup Probe
```python
status = checker.check_startup()
# Validates initialization complete
```

### Metrics Export
Health metrics available at `/metrics` endpoint:
- Uptime
- Success rate
- Component health
- Resource usage

## Error Handling

### Retry with Exponential Backoff
```python
from medassist.error_handling import retry

@retry(max_attempts=3, delay=1.0, backoff=2.0)
def call_external_api():
    # Will retry up to 3 times with exponential backoff
    pass
```

### Circuit Breaker
Prevents cascading failures:
```python
from medassist.error_handling import CircuitBreaker

circuit = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60
)

result = circuit.call(unreliable_function)
```

States:
- **CLOSED**: Normal operation
- **OPEN**: Failing, rejecting requests
- **HALF_OPEN**: Testing recovery

### Graceful Degradation
System continues with reduced functionality:
```python
from medassist.error_handling import GracefulDegradation

degradation = GracefulDegradation()

# Mark feature as degraded
degradation.mark_degraded("cloud_sync", exception)

# Check if degraded
if degradation.is_degraded("cloud_sync"):
    # Use local fallback
    pass
```

### Error Recovery Strategies
Automatic recovery from common failures:
- **Model Error**: Use rule-based fallback
- **Database Error**: Use cached data
- **Cloud Error**: Switch to offline mode

## Testing

### Test Coverage
Comprehensive test suite with 80%+ coverage:

#### Unit Tests
```bash
pytest tests/test_core.py -v --cov=medassist
```

Tests:
- Semantic routing
- Confidence tracking
- Model selection
- RAG system
- Circuit breaker
- Retry logic

#### Integration Tests
```bash
pytest tests/test_integration.py -v
```

Tests:
- Rural deployment scenarios
- Hybrid mode switching
- Error recovery
- Performance requirements
- Security features

### Running Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=medassist --cov-report=html

# Specific test file
pytest tests/test_core.py -v

# Specific test function
pytest tests/test_core.py::TestSemanticRouter::test_route_history_query -v
```

## CI/CD Pipeline

### GitHub Actions Workflow
Automated pipeline on every push/PR:

#### 1. Test Stage
- Run unit tests (Python 3.9, 3.10, 3.11)
- Run integration tests
- Generate coverage reports
- Upload to Codecov

#### 2. Lint Stage
- flake8 (code quality)
- black (code formatting)
- isort (import sorting)
- mypy (type checking)

#### 3. Security Stage
- bandit (security vulnerabilities)
- safety (dependency vulnerabilities)

#### 4. Build Stage
- Build Python package
- Upload artifacts

#### 5. Docker Stage
- Build Docker image
- Push to Docker Hub (on main branch)

### Running Locally
```bash
# Run linting
flake8 medassist/
black medassist/
isort medassist/

# Run security checks
bandit -r medassist/
safety check

# Build Docker image
docker build -t medassist:latest .
```

## Deployment

### Docker Deployment

#### Single Container
```bash
docker run -d \
  --name medassist \
  -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e MEDASSIST_ENV=production \
  -e CLINIC_ID=clinic_001 \
  medassist:latest
```

#### Docker Compose
```bash
# Start services
docker-compose up -d

# With monitoring
docker-compose --profile monitoring up -d

# View logs
docker-compose logs -f medassist

# Health check
docker-compose exec medassist python -c "from medassist.health_checks import HealthChecker; print(HealthChecker().check_readiness())"
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medassist
spec:
  replicas: 3
  selector:
    matchLabels:
      app: medassist
  template:
    metadata:
      labels:
        app: medassist
    spec:
      containers:
      - name: medassist
        image: medassist:latest
        ports:
        - containerPort: 7860
        - containerPort: 9090
        env:
        - name: MEDASSIST_ENV
          value: "production"
        livenessProbe:
          httpGet:
            path: /health/liveness
            port: 7860
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 7860
          initialDelaySeconds: 30
          periodSeconds: 10
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

### Environment-Specific Deployments

#### Development
```bash
export MEDASSIST_ENV=development
python app.py
```

#### Staging
```bash
export MEDASSIST_ENV=staging
python app.py
```

#### Production
```bash
export MEDASSIST_ENV=production
python app.py
```

## Security

### Encryption
- **Data at Rest**: AES-256 encryption
- **Data in Transit**: TLS 1.3
- **Key Management**: Secure key storage

### Authentication
```python
# Enable authentication
config.security.enable_auth = True
config.security.session_timeout_minutes = 60
config.security.max_login_attempts = 5
```

### Audit Trail
All security-sensitive operations logged:
- User authentication
- Data access
- Configuration changes
- Failed login attempts

### Security Best Practices
1. **Principle of Least Privilege**: Minimal permissions
2. **Defense in Depth**: Multiple security layers
3. **Zero Trust**: Verify everything
4. **Regular Updates**: Keep dependencies updated
5. **Secrets Management**: Never commit secrets

### Security Scanning
```bash
# Dependency vulnerabilities
safety check

# Code vulnerabilities
bandit -r medassist/

# Docker image scanning
docker scan medassist:latest
```

## Performance

### Resource Limits
```python
# Configuration
config.performance.max_concurrent_requests = 50
config.performance.request_timeout_seconds = 60
config.performance.thread_pool_size = 8
```

### Caching
```python
# Enable caching
config.performance.enable_caching = True
config.performance.cache_ttl_seconds = 3600
```

### Performance Targets
- **Local Inference**: <10s per query
- **Cloud API**: <500ms per request
- **Routing**: <100ms
- **Health Check**: <50ms

### Performance Monitoring
```python
import time

start = time.time()
result = process_query(query)
latency = time.time() - start

logger.info(f"Query processed", extra={
    "extra_fields": {
        "latency_ms": latency * 1000,
        "query_length": len(query)
    }
})
```

## Observability

### Three Pillars

#### 1. Logs
- Structured JSON logging
- Log aggregation (ELK, Splunk)
- Log levels and filtering
- Audit trails

#### 2. Metrics
- Health check metrics
- Performance metrics
- Resource utilization
- Custom business metrics

#### 3. Traces
- Request tracing
- Distributed tracing (future)
- Performance profiling

### Monitoring Stack

#### Prometheus + Grafana
```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana
open http://localhost:3000
```

#### Key Metrics
- **Availability**: Uptime, success rate
- **Performance**: Latency, throughput
- **Errors**: Error rate, failure types
- **Resources**: CPU, memory, disk

### Alerting
Configure alerts for:
- Service downtime
- High error rate
- Resource exhaustion
- Security incidents

## Conclusion

These production best practices ensure MedAssist is:
- ✅ **Reliable**: High availability, graceful degradation
- ✅ **Secure**: Encryption, authentication, audit logging
- ✅ **Performant**: Optimized, cached, monitored
- ✅ **Maintainable**: Well-tested, documented, automated
- ✅ **Observable**: Comprehensive logging, metrics, health checks
- ✅ **Scalable**: Horizontal scaling, resource limits

For detailed architecture, see [DEPLOYMENT_ARCHITECTURE.md](DEPLOYMENT_ARCHITECTURE.md).
