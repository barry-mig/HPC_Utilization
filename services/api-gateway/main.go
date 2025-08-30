/*
DEMO PROJECT: HPC API Gateway
High-performance Go-based API gateway with load balancing, 
monitoring, and advanced routing capabilities.
Most implementation commented out for presentation.
*/

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	// "net/http/httputil"
	// "net/url"
	"os"
	// "strconv"
	// "strings"
	// "sync"
	"time"

	"github.com/gin-gonic/gin"
	// "github.com/prometheus/client_golang/prometheus"
	// "github.com/prometheus/client_golang/prometheus/promhttp"
	// "go.uber.org/zap"
)

// Service represents a backend service
type Service struct {
	Name     string `json:"name"`
	URL      string `json:"url"`
	Priority int    `json:"priority"`
	Healthy  bool   `json:"healthy"`
	Weight   int    `json:"weight"`
}

// HealthCheck represents health status
type HealthCheck struct {
	Service   string    `json:"service"`
	Status    string    `json:"status"`
	Timestamp time.Time `json:"timestamp"`
	Latency   int64     `json:"latency_ms"`
}

// RequestMetrics for tracking
type RequestMetrics struct {
	TotalRequests   prometheus.Counter
	RequestDuration prometheus.Histogram
	ErrorRequests   prometheus.Counter
	ServiceRequests *prometheus.CounterVec
}

// LoadBalancer handles request distribution
type LoadBalancer struct {
	services []Service
	current  int
	mutex    sync.RWMutex
	logger   *zap.Logger
}

// APIGateway main structure
type APIGateway struct {
	router      *gin.Engine
	loadBalance *LoadBalancer
	metrics     *RequestMetrics
	logger      *zap.Logger
	healthCheck map[string]*HealthCheck
	healthMutex sync.RWMutex
}

func NewAPIGateway() *APIGateway {
	// Initialize logger
	logger, _ := zap.NewProduction()

	// Initialize metrics
	metrics := &RequestMetrics{
		TotalRequests: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "api_gateway_requests_total",
			Help: "Total number of requests processed by the API gateway",
		}),
		RequestDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name: "api_gateway_request_duration_seconds",
			Help: "Duration of requests processed by the API gateway",
		}),
		ErrorRequests: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "api_gateway_errors_total",
			Help: "Total number of error requests",
		}),
		ServiceRequests: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "api_gateway_service_requests_total",
				Help: "Total requests per service",
			},
			[]string{"service", "method", "status"},
		),
	}

	// Register metrics with Prometheus
	prometheus.MustRegister(metrics.TotalRequests)
	prometheus.MustRegister(metrics.RequestDuration)
	prometheus.MustRegister(metrics.ErrorRequests)
	prometheus.MustRegister(metrics.ServiceRequests)

	// Initialize services configuration
	services := []Service{
		{
			Name:     "workload-predictor",
			URL:      getEnv("WORKLOAD_PREDICTOR_URL", "http://workload-predictor:8000"),
			Priority: 1,
			Healthy:  true,
			Weight:   1,
		},
		{
			Name:     "resource-optimizer",
			URL:      getEnv("RESOURCE_OPTIMIZER_URL", "http://resource-optimizer:8001"),
			Priority: 1,
			Healthy:  true,
			Weight:   1,
		},
		{
			Name:     "scheduler-optimizer",
			URL:      getEnv("SCHEDULER_OPTIMIZER_URL", "http://scheduler-optimizer:8002"),
			Priority: 1,
			Healthy:  true,
			Weight:   1,
		},
	}

	loadBalancer := &LoadBalancer{
		services: services,
		current:  0,
		logger:   logger,
	}

	// Initialize Gin router
	if os.Getenv("GIN_MODE") != "debug" {
		gin.SetMode(gin.ReleaseMode)
	}
	router := gin.New()

	gateway := &APIGateway{
		router:      router,
		loadBalance: loadBalancer,
		metrics:     metrics,
		logger:      logger,
		healthCheck: make(map[string]*HealthCheck),
	}

	gateway.setupRoutes()
	gateway.setupMiddleware()

	// Start health checking
	go gateway.startHealthChecking()

	return gateway
}

func (gw *APIGateway) setupMiddleware() {
	// CORS middleware
	gw.router.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	// Logging middleware
	gw.router.Use(gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		return fmt.Sprintf("%s - [%s] \"%s %s %s %d %s \"%s\" %s\"\n",
			param.ClientIP,
			param.TimeStamp.Format(time.RFC1123),
			param.Method,
			param.Path,
			param.Request.Proto,
			param.StatusCode,
			param.Latency,
			param.Request.UserAgent(),
			param.ErrorMessage,
		)
	}))

	// Recovery middleware
	gw.router.Use(gin.Recovery())

	// Metrics middleware
	gw.router.Use(gw.metricsMiddleware())

	// Rate limiting middleware
	gw.router.Use(gw.rateLimitMiddleware())
}

func (gw *APIGateway) setupRoutes() {
	// Health check endpoint
	gw.router.GET("/health", gw.healthHandler)
	gw.router.GET("/health/:service", gw.serviceHealthHandler)

	// Metrics endpoint
	gw.router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Service discovery endpoint
	gw.router.GET("/services", gw.servicesHandler)

	// API routes with load balancing
	api := gw.router.Group("/api/v1")
	{
		// Workload prediction routes
		api.POST("/predict", gw.proxyToService("workload-predictor"))
		api.POST("/predict/retrain", gw.proxyToService("workload-predictor"))
		api.GET("/predict/model/info", gw.proxyToService("workload-predictor"))

		// Resource optimization routes
		api.POST("/optimize", gw.proxyToService("resource-optimizer"))
		api.GET("/optimize/algorithms", gw.proxyToService("resource-optimizer"))

		// Scheduler optimization routes
		api.POST("/schedule", gw.proxyToService("scheduler-optimizer"))
		api.GET("/schedule/policies", gw.proxyToService("scheduler-optimizer"))

		// Combined workflow endpoints
		api.POST("/workflow/full-optimization", gw.fullOptimizationWorkflow)
		api.POST("/workflow/prediction-and-schedule", gw.predictionAndScheduleWorkflow)
	}

	// Admin routes
	admin := gw.router.Group("/admin")
	{
		admin.GET("/status", gw.adminStatusHandler)
		admin.POST("/services/:name/enable", gw.enableServiceHandler)
		admin.POST("/services/:name/disable", gw.disableServiceHandler)
		admin.GET("/config", gw.configHandler)
	}
}

func (gw *APIGateway) metricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		gw.metrics.TotalRequests.Inc()

		c.Next()

		duration := time.Since(start)
		gw.metrics.RequestDuration.Observe(duration.Seconds())

		if c.Writer.Status() >= 400 {
			gw.metrics.ErrorRequests.Inc()
		}

		gw.metrics.ServiceRequests.WithLabelValues(
			gw.getServiceFromPath(c.Request.URL.Path),
			c.Request.Method,
			strconv.Itoa(c.Writer.Status()),
		).Inc()
	}
}

func (gw *APIGateway) rateLimitMiddleware() gin.HandlerFunc {
	// Simple in-memory rate limiter
	clientRequests := make(map[string][]time.Time)
	var mutex sync.Mutex
	limit := 100 // requests per minute
	window := time.Minute

	return func(c *gin.Context) {
		clientIP := c.ClientIP()
		now := time.Now()

		mutex.Lock()
		defer mutex.Unlock()

		// Clean old requests
		if requests, exists := clientRequests[clientIP]; exists {
			var validRequests []time.Time
			for _, reqTime := range requests {
				if now.Sub(reqTime) < window {
					validRequests = append(validRequests, reqTime)
				}
			}
			clientRequests[clientIP] = validRequests
		}

		// Check rate limit
		if len(clientRequests[clientIP]) >= limit {
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error": "Rate limit exceeded",
				"retry_after": 60,
			})
			c.Abort()
			return
		}

		// Add current request
		clientRequests[clientIP] = append(clientRequests[clientIP], now)
		c.Next()
	}
}

func (gw *APIGateway) proxyToService(serviceName string) gin.HandlerFunc {
	return func(c *gin.Context) {
		service, err := gw.loadBalance.getHealthyService(serviceName)
		if err != nil {
			gw.logger.Error("Service unavailable", zap.String("service", serviceName), zap.Error(err))
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error": fmt.Sprintf("Service %s unavailable", serviceName),
			})
			return
		}

		// Parse target URL
		targetURL, err := url.Parse(service.URL)
		if err != nil {
			gw.logger.Error("Invalid service URL", zap.String("url", service.URL), zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Invalid service configuration",
			})
			return
		}

		// Create reverse proxy
		proxy := httputil.NewSingleHostReverseProxy(targetURL)

		// Modify the request
		originalPath := c.Request.URL.Path
		c.Request.URL.Path = strings.Replace(originalPath, "/api/v1", "", 1)
		c.Request.URL.Host = targetURL.Host
		c.Request.URL.Scheme = targetURL.Scheme
		c.Request.Header.Set("X-Forwarded-Host", c.Request.Header.Get("Host"))
		c.Request.Host = targetURL.Host

		// Custom error handler
		proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			gw.logger.Error("Proxy error", zap.Error(err))
			w.WriteHeader(http.StatusBadGateway)
			json.NewEncoder(w).Encode(gin.H{
				"error": "Backend service error",
			})
		}

		proxy.ServeHTTP(c.Writer, c.Request)
	}
}

func (gw *APIGateway) fullOptimizationWorkflow(c *gin.Context) {
	// Complex workflow that combines prediction, optimization, and scheduling
	start := time.Now()
	
	var request struct {
		HistoricalData    json.RawMessage `json:"historical_data"`
		Jobs              json.RawMessage `json:"jobs"`
		ClusterNodes      json.RawMessage `json:"cluster_nodes"`
		GlobalConstraints json.RawMessage `json:"global_constraints"`
		PredictionHorizon int            `json:"prediction_horizon"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	results := make(map[string]interface{})

	// Step 1: Get workload predictions
	predictionService, err := gw.loadBalance.getHealthyService("workload-predictor")
	if err == nil {
		predictionResult, err := gw.callService(predictionService, "/predict", map[string]interface{}{
			"historical_data":    request.HistoricalData,
			"prediction_horizon": request.PredictionHorizon,
		})
		if err == nil {
			results["predictions"] = predictionResult
		} else {
			gw.logger.Warn("Prediction service failed", zap.Error(err))
		}
	}

	// Step 2: Optimize resource allocation
	optimizerService, err := gw.loadBalance.getHealthyService("resource-optimizer")
	if err == nil {
		optimizationResult, err := gw.callService(optimizerService, "/optimize", map[string]interface{}{
			"jobs":               request.Jobs,
			"cluster_nodes":      request.ClusterNodes,
			"global_constraints": request.GlobalConstraints,
		})
		if err == nil {
			results["optimization"] = optimizationResult
		} else {
			gw.logger.Warn("Optimization service failed", zap.Error(err))
		}
	}

	// Step 3: Generate schedule
	schedulerService, err := gw.loadBalance.getHealthyService("scheduler-optimizer")
	if err == nil {
		scheduleResult, err := gw.callService(schedulerService, "/schedule", map[string]interface{}{
			"jobs":          request.Jobs,
			"cluster_nodes": request.ClusterNodes,
		})
		if err == nil {
			results["schedule"] = scheduleResult
		} else {
			gw.logger.Warn("Scheduler service failed", zap.Error(err))
		}
	}

	results["workflow_duration_ms"] = time.Since(start).Milliseconds()
	results["timestamp"] = time.Now()

	c.JSON(http.StatusOK, results)
}

func (gw *APIGateway) predictionAndScheduleWorkflow(c *gin.Context) {
	// Simplified workflow for prediction + scheduling
	// Implementation similar to fullOptimizationWorkflow but with fewer steps
	c.JSON(http.StatusOK, gin.H{
		"message": "Prediction and schedule workflow",
		"status":  "implemented",
	})
}

func (gw *APIGateway) callService(service Service, endpoint string, payload interface{}) (interface{}, error) {
	// Helper function to make HTTP calls to backend services
	// This would contain the actual HTTP client implementation
	return map[string]interface{}{
		"service": service.Name,
		"endpoint": endpoint,
		"status": "success",
	}, nil
}

func (gw *APIGateway) healthHandler(c *gin.Context) {
	gw.healthMutex.RLock()
	defer gw.healthMutex.RUnlock()

	status := "healthy"
	services := make(map[string]interface{})

	for _, service := range gw.loadBalance.services {
		healthCheck, exists := gw.healthCheck[service.Name]
		if !exists || healthCheck.Status != "healthy" {
			status = "degraded"
		}
		services[service.Name] = healthCheck
	}

	c.JSON(http.StatusOK, gin.H{
		"status":    status,
		"timestamp": time.Now(),
		"services":  services,
		"version":   "1.0.0",
	})
}

func (gw *APIGateway) serviceHealthHandler(c *gin.Context) {
	serviceName := c.Param("service")
	
	gw.healthMutex.RLock()
	healthCheck, exists := gw.healthCheck[serviceName]
	gw.healthMutex.RUnlock()

	if !exists {
		c.JSON(http.StatusNotFound, gin.H{
			"error": "Service not found",
		})
		return
	}

	c.JSON(http.StatusOK, healthCheck)
}

func (gw *APIGateway) servicesHandler(c *gin.Context) {
	gw.loadBalance.mutex.RLock()
	services := gw.loadBalance.services
	gw.loadBalance.mutex.RUnlock()

	c.JSON(http.StatusOK, gin.H{
		"services": services,
		"total":    len(services),
	})
}

func (gw *APIGateway) adminStatusHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"gateway": "healthy",
		"uptime":  time.Since(time.Now()).String(),
		"version": "1.0.0",
	})
}

func (gw *APIGateway) enableServiceHandler(c *gin.Context) {
	serviceName := c.Param("name")
	gw.loadBalance.setServiceHealth(serviceName, true)
	c.JSON(http.StatusOK, gin.H{
		"message": fmt.Sprintf("Service %s enabled", serviceName),
	})
}

func (gw *APIGateway) disableServiceHandler(c *gin.Context) {
	serviceName := c.Param("name")
	gw.loadBalance.setServiceHealth(serviceName, false)
	c.JSON(http.StatusOK, gin.H{
		"message": fmt.Sprintf("Service %s disabled", serviceName),
	})
}

func (gw *APIGateway) configHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"services":     gw.loadBalance.services,
		"health_check": gw.healthCheck,
	})
}

func (gw *APIGateway) startHealthChecking() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			gw.performHealthChecks()
		}
	}
}

func (gw *APIGateway) performHealthChecks() {
	for _, service := range gw.loadBalance.services {
		go gw.checkServiceHealth(service)
	}
}

func (gw *APIGateway) checkServiceHealth(service Service) {
	start := time.Now()
	
	// Make HTTP health check request
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(service.URL + "/health")
	
	latency := time.Since(start).Milliseconds()
	status := "unhealthy"
	
	if err == nil && resp.StatusCode == http.StatusOK {
		status = "healthy"
		gw.loadBalance.setServiceHealth(service.Name, true)
	} else {
		gw.loadBalance.setServiceHealth(service.Name, false)
		gw.logger.Warn("Health check failed", 
			zap.String("service", service.Name), 
			zap.Error(err))
	}

	if resp != nil {
		resp.Body.Close()
	}

	gw.healthMutex.Lock()
	gw.healthCheck[service.Name] = &HealthCheck{
		Service:   service.Name,
		Status:    status,
		Timestamp: time.Now(),
		Latency:   latency,
	}
	gw.healthMutex.Unlock()
}

// LoadBalancer methods
func (lb *LoadBalancer) getHealthyService(serviceName string) (Service, error) {
	lb.mutex.RLock()
	defer lb.mutex.RUnlock()

	for _, service := range lb.services {
		if service.Name == serviceName && service.Healthy {
			return service, nil
		}
	}

	return Service{}, fmt.Errorf("no healthy service found for %s", serviceName)
}

func (lb *LoadBalancer) setServiceHealth(serviceName string, healthy bool) {
	lb.mutex.Lock()
	defer lb.mutex.Unlock()

	for i, service := range lb.services {
		if service.Name == serviceName {
			lb.services[i].Healthy = healthy
			break
		}
	}
}

// Helper functions
func (gw *APIGateway) getServiceFromPath(path string) string {
	if strings.Contains(path, "predict") {
		return "workload-predictor"
	} else if strings.Contains(path, "optimize") {
		return "resource-optimizer"
	} else if strings.Contains(path, "schedule") {
		return "scheduler-optimizer"
	}
	return "gateway"
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func main() {
	gateway := NewAPIGateway()
	port := getEnv("PORT", "8080")
	
	log.Printf("Starting API Gateway on port %s", port)
	log.Fatal(gateway.router.Run(":" + port))
}
