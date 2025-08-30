# Development Guide

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** - For ML services
- **Go 1.21+** - For high-performance API Gateway
- **Docker** - For containerization
- **Kubernetes** - For orchestration (minikube for local development)
- **kubectl** - Kubernetes CLI
- **Helm** - Kubernetes package manager

### Quick Setup

```bash
# Clone and setup
git clone <repository-url>
cd HPC_Utilization
make setup

# Start local development
make run-local

# Or run complete demo
make demo
```

## 🏗️ Architecture Deep Dive

### Service Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           API Gateway (Go)                      │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │  Load Balancer  │ │   Rate Limiter  │ │    Metrics      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼───────┐     ┌────────▼────────┐     ┌────────▼────────┐
│   Workload    │     │    Resource     │     │   Scheduler     │
│  Predictor    │     │   Optimizer     │     │   Optimizer     │
│   (Python)    │     │   (Python)      │     │   (Python)      │
│               │     │                 │     │                 │
│ • Time Series │     │ • Optimization  │     │ • Job Scheduling│
│ • ML Models   │     │ • Algorithms    │     │ • Policies      │
│ • Forecasting │     │ • Resource Mgmt │     │ • Priority Mgmt │
└───────────────┘     └─────────────────┘     └─────────────────┘
```

### Data Flow

1. **Historical Data Ingestion** → Workload Predictor
2. **Prediction Generation** → Resource Optimizer
3. **Resource Allocation** → Scheduler Optimizer
4. **Job Scheduling** → HPC Cluster

### Technology Stack Details

#### Backend Services
- **FastAPI** - High-performance Python web framework
- **Pydantic** - Data validation and serialization
- **scikit-learn** - Machine learning algorithms
- **pandas/numpy** - Data processing
- **scipy** - Scientific computing for optimization

#### API Gateway (Go)
- **Gin** - High-performance HTTP web framework
- **Prometheus** - Metrics collection
- **Zap** - High-performance logging
- **Built-in Load Balancing** - Round-robin with health checks

#### Infrastructure
- **Kubernetes** - Container orchestration
- **Docker** - Containerization
- **Prometheus + Grafana** - Monitoring and alerting
- **GitHub Actions** - CI/CD pipeline

## 🔧 Development Workflow

### 1. Local Development Setup

```bash
# Setup Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup Go environment
cd services/api-gateway
go mod download

# Start dependencies
docker-compose -f docker-compose.dev.yml up -d

# Run services
make run-local
```

### 2. Service Development

#### Adding New Features to Workload Predictor

1. **Model Development**:
   ```python
   # Add new model in models/time_series/
   class NewPredictionModel:
       def train(self, data):
           # Implementation
           pass
       
       def predict(self, features):
           # Implementation
           pass
   ```

2. **API Endpoint**:
   ```python
   # Add to services/workload-predictor/main.py
   @app.post("/new-endpoint")
   async def new_prediction_endpoint(request: NewRequest):
       # Implementation
       pass
   ```

3. **Tests**:
   ```python
   # Add to tests/unit/test_workload_predictor.py
   def test_new_endpoint(self, client):
       response = client.post("/new-endpoint", json={})
       assert response.status_code == 200
   ```

#### Adding New Optimization Algorithms

1. **Algorithm Implementation**:
   ```python
   # Add to services/resource-optimizer/main.py
   def _optimize_with_new_algorithm(self, request):
       # New optimization logic
       pass
   ```

2. **Register Algorithm**:
   ```python
   self.optimization_methods["new_algorithm"] = self._optimize_with_new_algorithm
   ```

### 3. Testing Strategy

#### Unit Tests
```bash
# Run all unit tests
make test-unit

# Run specific service tests
cd services/workload-predictor
python -m pytest tests/ -v --cov=.
```

#### Integration Tests
```bash
# Test service interactions
make test-integration

# Test with real Kubernetes cluster
make deploy
make verify-deployment
```

#### Load Testing
```bash
# Install k6
# Run load tests
make test-load
```

### 4. Performance Optimization

#### Python Services
- Use **async/await** for I/O operations
- Implement **caching** for expensive computations
- **Batch processing** for multiple predictions
- **Connection pooling** for database access

#### Go API Gateway
- **Goroutines** for concurrent request handling
- **Connection pooling** for backend services
- **Circuit breakers** for fault tolerance
- **Metrics collection** for performance monitoring

## 🚀 Deployment Guide

### Local Kubernetes (minikube)

```bash
# Start minikube
minikube start --memory=8192 --cpus=4

# Deploy services
make deploy

# Access services
kubectl port-forward service/api-gateway-service 8080:80 -n hpc-system
```

### Production Deployment

```bash
# Build and push images
docker login ghcr.io
make build
docker push ghcr.io/your-org/hpc-workload-predictor:latest
docker push ghcr.io/your-org/hpc-resource-optimizer:latest
docker push ghcr.io/your-org/hpc-api-gateway:latest

# Deploy to production cluster
kubectl config use-context production
make deploy
```

### Environment Configuration

#### Development
```yaml
# kubernetes/configmaps/dev-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: hpc-system
data:
  log_level: "DEBUG"
  metrics_enabled: "true"
  model_update_interval: "1h"
```

#### Production
```yaml
# kubernetes/configmaps/prod-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: hpc-system
data:
  log_level: "INFO"
  metrics_enabled: "true"
  model_update_interval: "24h"
```

## 📊 Monitoring and Observability

### Metrics Collection

#### Application Metrics
- **Request latency** - API response times
- **Prediction accuracy** - Model performance
- **Resource utilization** - System usage
- **Error rates** - Service health

#### Infrastructure Metrics
- **Pod CPU/Memory** - Resource consumption
- **Network traffic** - Service communication
- **Storage usage** - Persistent volume usage

### Logging Strategy

#### Structured Logging
```python
import structlog

logger = structlog.get_logger()
logger.info("Prediction completed", 
           prediction_id="123",
           accuracy=0.95,
           latency_ms=150)
```

#### Log Aggregation
- **Centralized logging** with ELK stack or similar
- **Log correlation** across services
- **Alert configuration** for error patterns

### Alerting Rules

```yaml
# Alert on high prediction latency
- alert: HighPredictionLatency
  expr: histogram_quantile(0.95, workload_prediction_duration_seconds_bucket) > 5
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "High prediction latency detected"

# Alert on service down
- alert: ServiceDown
  expr: up{job="workload-predictor"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Workload predictor service is down"
```

## 🔒 Security Best Practices

### Container Security
- **Non-root users** in containers
- **Read-only root filesystems**
- **Minimal base images** (Alpine Linux)
- **Security scanning** with Trivy

### Kubernetes Security
- **RBAC** for service accounts
- **Network policies** for traffic isolation
- **Pod security policies**
- **Secret management** with sealed-secrets

### API Security
- **Rate limiting** to prevent abuse
- **Input validation** with Pydantic
- **CORS configuration** for web access
- **Authentication/Authorization** (implement JWT tokens)

## 🐛 Troubleshooting

### Common Issues

#### Service Not Starting
```bash
# Check pod status
kubectl get pods -n hpc-system

# Check logs
kubectl logs deployment/workload-predictor -n hpc-system

# Check events
kubectl describe pod <pod-name> -n hpc-system
```

#### Poor Prediction Performance
1. **Check training data quality**
2. **Verify feature engineering**
3. **Monitor model metrics**
4. **Consider model retraining**

#### High Memory Usage
1. **Optimize batch sizes**
2. **Implement data streaming**
3. **Add memory limits to pods**
4. **Monitor memory leaks**

### Debugging Tools

#### Local Development
```bash
# Python debugger
import pdb; pdb.set_trace()

# Go debugger (delve)
dlv debug main.go
```

#### Production Debugging
```bash
# Access pod shell
kubectl exec -it deployment/api-gateway -n hpc-system -- /bin/sh

# Port forward for debugging
kubectl port-forward pod/<pod-name> 8080:8080 -n hpc-system
```

## 📈 Performance Tuning

### Python Services
- **Async programming** for I/O bound operations
- **Multiprocessing** for CPU bound tasks
- **Memory profiling** with memory_profiler
- **Performance profiling** with cProfile

### Go Services
- **Goroutine optimization**
- **Memory pooling** for frequent allocations
- **Profiling with pprof**
- **Garbage collection tuning**

### Kubernetes Optimization
- **Resource requests/limits** optimization
- **Horizontal Pod Autoscaling** configuration
- **Node affinity** for performance-critical pods
- **Persistent volume** optimization

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

1. **Code Quality Checks**
   - Unit tests
   - Security scanning
   - Code linting

2. **Build Phase**
   - Docker image building
   - Multi-architecture builds
   - Image scanning

3. **Deployment Phase**
   - Staging deployment
   - Integration tests
   - Production deployment

4. **Post-Deployment**
   - Health checks
   - Performance tests
   - Monitoring verification

### Release Process

1. **Feature Development** → `feature/*` branch
2. **Pull Request** → Code review and tests
3. **Merge to `develop`** → Staging deployment
4. **Merge to `main`** → Production deployment
5. **Git Tag** → Release creation

## 📚 Additional Resources

### Learning Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Go Programming Language](https://golang.org/doc/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Tools and Extensions
- **VS Code Extensions**: Python, Go, Kubernetes, Docker
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Development**: Postman, k9s, kubectx
- **Testing**: pytest, testify, k6

### Community
- **Kubernetes Slack**: #kubernetes-users
- **FastAPI Discord**: FastAPI community
- **Go Community**: golang-nuts mailing list
