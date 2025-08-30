# HPC Utilization System Makefile
# Comprehensive build, test, and deployment automation

# Configuration
SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help setup clean test build run deploy monitor

# Variables
PROJECT_NAME := hpc-utilization
DOCKER_REGISTRY := ghcr.io
NAMESPACE := hpc-system
SERVICES := workload-predictor resource-optimizer api-gateway
PYTHON_VERSION := 3.11
GO_VERSION := 1.21

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(BLUE)HPC Utilization System - Available Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Services:$(NC) $(SERVICES)"
	@echo "$(YELLOW)Namespace:$(NC) $(NAMESPACE)"

##@ Setup and Installation

setup: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@$(MAKE) setup-python
	@$(MAKE) setup-go
	@$(MAKE) setup-kubernetes
	@$(MAKE) install-dependencies
	@echo "$(GREEN)✅ Development environment setup complete!$(NC)"

setup-python: ## Setup Python environment
	@echo "$(YELLOW)Setting up Python environment...$(NC)"
	@command -v python$(PYTHON_VERSION) >/dev/null 2>&1 || { \
		echo "$(RED)❌ Python $(PYTHON_VERSION) not found. Please install it first.$(NC)"; \
		exit 1; \
	}
	@python$(PYTHON_VERSION) -m venv venv
	@source venv/bin/activate && pip install --upgrade pip

setup-go: ## Setup Go environment
	@echo "$(YELLOW)Setting up Go environment...$(NC)"
	@command -v go >/dev/null 2>&1 || { \
		echo "$(RED)❌ Go not found. Please install Go $(GO_VERSION) first.$(NC)"; \
		exit 1; \
	}
	@cd services/api-gateway && go mod download

setup-kubernetes: ## Setup Kubernetes tools
	@echo "$(YELLOW)Setting up Kubernetes tools...$(NC)"
	@command -v kubectl >/dev/null 2>&1 || { \
		echo "$(RED)❌ kubectl not found. Please install kubectl first.$(NC)"; \
		exit 1; \
	}
	@command -v helm >/dev/null 2>&1 || { \
		echo "$(YELLOW)⚠️  Helm not found. Installing...$(NC)"; \
		curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash; \
	}

install-dependencies: ## Install all project dependencies
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	@for service in workload-predictor resource-optimizer; do \
		echo "Installing Python dependencies for $$service..."; \
		source venv/bin/activate && pip install -r services/$$service/requirements.txt; \
	done
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

##@ Testing

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	@$(MAKE) test-unit
	@$(MAKE) test-integration
	@$(MAKE) test-security
	@echo "$(GREEN)✅ All tests completed!$(NC)"

test-unit: ## Run unit tests
	@echo "$(YELLOW)Running unit tests...$(NC)"
	@source venv/bin/activate && \
		cd tests/unit && \
		python -m pytest . -v --cov=../../services --cov-report=html --cov-report=term
	@cd services/api-gateway && go test -v -race -coverprofile=coverage.out ./...
	@echo "$(GREEN)✅ Unit tests completed$(NC)"

test-integration: ## Run integration tests
	@echo "$(YELLOW)Running integration tests...$(NC)"
	@source venv/bin/activate && \
		cd tests/integration && \
		python -m pytest . -v
	@echo "$(GREEN)✅ Integration tests completed$(NC)"

test-security: ## Run security tests
	@echo "$(YELLOW)Running security tests...$(NC)"
	@source venv/bin/activate && bandit -r services/ -f json -o security-report.json || true
	@cd services/api-gateway && gosec ./... || true
	@echo "$(GREEN)✅ Security tests completed$(NC)"

test-load: ## Run load tests
	@echo "$(YELLOW)Running load tests...$(NC)"
	@command -v k6 >/dev/null 2>&1 || { \
		echo "$(RED)❌ k6 not found. Please install k6 first.$(NC)"; \
		exit 1; \
	}
	@k6 run tests/load/load-test.js
	@echo "$(GREEN)✅ Load tests completed$(NC)"

##@ Building

build: ## Build all services
	@echo "$(BLUE)Building all services...$(NC)"
	@$(MAKE) build-docker
	@echo "$(GREEN)✅ All services built!$(NC)"

build-docker: ## Build Docker images for all services
	@echo "$(YELLOW)Building Docker images...$(NC)"
	@for service in $(SERVICES); do \
		echo "Building $$service..."; \
		cd services/$$service && \
		docker build -t $(PROJECT_NAME)/$$service:latest . && \
		docker tag $(PROJECT_NAME)/$$service:latest $(DOCKER_REGISTRY)/$(PROJECT_NAME)/$$service:latest; \
		cd ../..; \
	done
	@echo "$(GREEN)✅ Docker images built$(NC)"

build-service: ## Build specific service (usage: make build-service SERVICE=workload-predictor)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)❌ Please specify SERVICE variable$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Building $(SERVICE)...$(NC)"
	@cd services/$(SERVICE) && docker build -t $(PROJECT_NAME)/$(SERVICE):latest .
	@echo "$(GREEN)✅ $(SERVICE) built$(NC)"

##@ Development

run-local: ## Run services locally
	@echo "$(BLUE)Starting services locally...$(NC)"
	@$(MAKE) run-dependencies
	@$(MAKE) run-services
	@echo "$(GREEN)✅ Services running locally!$(NC)"

run-dependencies: ## Start local dependencies (Redis, PostgreSQL)
	@echo "$(YELLOW)Starting dependencies...$(NC)"
	@docker-compose -f docker-compose.dev.yml up -d postgres redis
	@echo "$(GREEN)✅ Dependencies started$(NC)"

run-services: ## Start all services locally
	@echo "$(YELLOW)Starting services...$(NC)"
	@trap 'kill 0' SIGINT; \
	source venv/bin/activate && \
	cd services/workload-predictor && python main.py & \
	cd services/resource-optimizer && python main.py & \
	cd services/api-gateway && go run main.go & \
	wait

run-service: ## Run specific service (usage: make run-service SERVICE=workload-predictor)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)❌ Please specify SERVICE variable$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Running $(SERVICE)...$(NC)"
	@if [ "$(SERVICE)" = "api-gateway" ]; then \
		cd services/$(SERVICE) && go run main.go; \
	else \
		source venv/bin/activate && cd services/$(SERVICE) && python main.py; \
	fi

##@ Deployment

deploy: ## Deploy to Kubernetes
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	@$(MAKE) deploy-namespace
	@$(MAKE) deploy-services
	@$(MAKE) deploy-monitoring
	@$(MAKE) verify-deployment
	@echo "$(GREEN)✅ Deployment completed!$(NC)"

deploy-namespace: ## Create namespace and configurations
	@echo "$(YELLOW)Creating namespace and configurations...$(NC)"
	@kubectl apply -f kubernetes/namespace-and-config.yaml
	@echo "$(GREEN)✅ Namespace created$(NC)"

deploy-services: ## Deploy all services
	@echo "$(YELLOW)Deploying services...$(NC)"
	@kubectl apply -f kubernetes/deployments/
	@echo "$(GREEN)✅ Services deployed$(NC)"

deploy-monitoring: ## Deploy monitoring stack
	@echo "$(YELLOW)Deploying monitoring...$(NC)"
	@kubectl apply -f kubernetes/monitoring/
	@echo "$(GREEN)✅ Monitoring deployed$(NC)"

verify-deployment: ## Verify deployment status
	@echo "$(YELLOW)Verifying deployment...$(NC)"
	@kubectl wait --for=condition=available --timeout=300s deployment --all -n $(NAMESPACE)
	@kubectl get pods -n $(NAMESPACE)
	@echo "$(GREEN)✅ Deployment verified$(NC)"

##@ Kubernetes Operations

k8s-status: ## Check Kubernetes deployment status
	@echo "$(BLUE)Kubernetes Status$(NC)"
	@kubectl get all -n $(NAMESPACE)

k8s-logs: ## View logs (usage: make k8s-logs SERVICE=workload-predictor)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)❌ Please specify SERVICE variable$(NC)"; \
		exit 1; \
	fi
	@kubectl logs -f deployment/$(SERVICE) -n $(NAMESPACE)

k8s-shell: ## Get shell access to pod (usage: make k8s-shell SERVICE=workload-predictor)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)❌ Please specify SERVICE variable$(NC)"; \
		exit 1; \
	fi
	@kubectl exec -it deployment/$(SERVICE) -n $(NAMESPACE) -- /bin/bash

k8s-port-forward: ## Port forward service (usage: make k8s-port-forward SERVICE=api-gateway PORT=8080)
	@if [ -z "$(SERVICE)" ] || [ -z "$(PORT)" ]; then \
		echo "$(RED)❌ Please specify SERVICE and PORT variables$(NC)"; \
		exit 1; \
	fi
	@kubectl port-forward service/$(SERVICE)-service $(PORT):$(PORT) -n $(NAMESPACE)

##@ Monitoring

monitor: ## Open monitoring dashboards
	@echo "$(BLUE)Opening monitoring dashboards...$(NC)"
	@$(MAKE) monitor-grafana &
	@$(MAKE) monitor-prometheus &
	@echo "$(GREEN)✅ Monitoring dashboards opened$(NC)"

monitor-grafana: ## Open Grafana dashboard
	@echo "$(YELLOW)Opening Grafana...$(NC)"
	@kubectl port-forward service/grafana-service 3000:3000 -n $(NAMESPACE) &
	@sleep 2
	@open http://localhost:3000 || echo "Visit http://localhost:3000"

monitor-prometheus: ## Open Prometheus dashboard
	@echo "$(YELLOW)Opening Prometheus...$(NC)"
	@kubectl port-forward service/prometheus-service 9090:9090 -n $(NAMESPACE) &
	@sleep 2
	@open http://localhost:9090 || echo "Visit http://localhost:9090"

##@ Data and Models

generate-data: ## Generate synthetic training data
	@echo "$(YELLOW)Generating synthetic data...$(NC)"
	@source venv/bin/activate && python models/time_series/workload_predictor.py
	@echo "$(GREEN)✅ Synthetic data generated$(NC)"

train-models: ## Train ML models
	@echo "$(YELLOW)Training ML models...$(NC)"
	@source venv/bin/activate && \
		cd models/time_series && \
		python workload_predictor.py
	@echo "$(GREEN)✅ Models trained$(NC)"

##@ Cleanup

clean: ## Clean up all resources
	@echo "$(BLUE)Cleaning up...$(NC)"
	@$(MAKE) clean-docker
	@$(MAKE) clean-kubernetes
	@$(MAKE) clean-local
	@echo "$(GREEN)✅ Cleanup completed!$(NC)"

clean-docker: ## Remove Docker images
	@echo "$(YELLOW)Removing Docker images...$(NC)"
	@for service in $(SERVICES); do \
		docker rmi $(PROJECT_NAME)/$$service:latest 2>/dev/null || true; \
	done
	@docker system prune -f
	@echo "$(GREEN)✅ Docker images cleaned$(NC)"

clean-kubernetes: ## Remove Kubernetes resources
	@echo "$(YELLOW)Removing Kubernetes resources...$(NC)"
	@kubectl delete namespace $(NAMESPACE) --ignore-not-found=true
	@echo "$(GREEN)✅ Kubernetes resources cleaned$(NC)"

clean-local: ## Clean local development files
	@echo "$(YELLOW)Cleaning local files...$(NC)"
	@rm -rf venv/
	@rm -rf **/__pycache__/
	@rm -rf **/*.pyc
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf *.log
	@rm -rf security-report.json
	@echo "$(GREEN)✅ Local files cleaned$(NC)"

##@ CI/CD

ci-test: ## Run CI tests (used in GitHub Actions)
	@echo "$(BLUE)Running CI tests...$(NC)"
	@$(MAKE) test-unit
	@$(MAKE) test-security
	@echo "$(GREEN)✅ CI tests completed$(NC)"

ci-build: ## Build for CI (used in GitHub Actions)
	@echo "$(BLUE)Building for CI...$(NC)"
	@$(MAKE) build-docker
	@echo "$(GREEN)✅ CI build completed$(NC)"

ci-deploy: ## Deploy for CI (used in GitHub Actions)
	@echo "$(BLUE)Deploying for CI...$(NC)"
	@$(MAKE) deploy
	@echo "$(GREEN)✅ CI deployment completed$(NC)"

##@ Utilities

format: ## Format code
	@echo "$(YELLOW)Formatting code...$(NC)"
	@source venv/bin/activate && \
		black services/ && \
		isort services/
	@cd services/api-gateway && go fmt ./...
	@echo "$(GREEN)✅ Code formatted$(NC)"

lint: ## Lint code
	@echo "$(YELLOW)Linting code...$(NC)"
	@source venv/bin/activate && \
		flake8 services/ && \
		pylint services/
	@cd services/api-gateway && golangci-lint run
	@echo "$(GREEN)✅ Code linted$(NC)"

docs: ## Generate documentation
	@echo "$(YELLOW)Generating documentation...$(NC)"
	@source venv/bin/activate && \
		cd docs && \
		make html
	@echo "$(GREEN)✅ Documentation generated$(NC)"

version: ## Show version information
	@echo "$(BLUE)Version Information$(NC)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Python: $(shell python --version 2>&1)"
	@echo "Go: $(shell go version 2>&1)"
	@echo "Docker: $(shell docker --version 2>&1)"
	@echo "Kubectl: $(shell kubectl version --client --short 2>&1)"

##@ Quick Actions

quick-start: ## Quick start for development
	@echo "$(BLUE)Quick starting development environment...$(NC)"
	@$(MAKE) setup
	@$(MAKE) run-local

demo: ## Run a complete demo
	@echo "$(BLUE)Running complete demo...$(NC)"
	@$(MAKE) generate-data
	@$(MAKE) build
	@$(MAKE) deploy
	@$(MAKE) verify-deployment
	@echo "$(GREEN)✅ Demo completed! Check the monitoring dashboards.$(NC)"

# Include any additional makefiles
-include Makefile.local
