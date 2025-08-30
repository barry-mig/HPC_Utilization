#!/usr/bin/env python3
"""
Demo test script for HPC Workload Predictor
"""
import requests
import json
from datetime import datetime, timedelta

def test_workload_predictor():
    """Test the workload predictor service"""
    base_url = "http://localhost:8000"
    
    print("🚀 Testing HPC Workload Predictor Demo")
    print("=" * 50)
    
    # Test health endpoint
    print("1. Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print(f"   Status: {health_data['status']}")
        print(f"   Model Loaded: {health_data['model_loaded']}")
        print(f"   Uptime: {health_data['uptime_seconds']:.2f} seconds")
        print("   ✅ Health check passed!")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return
    
    print("\n2. Testing Prediction Endpoint...")
    
    # Create sample historical data
    current_time = datetime.now()
    historical_data = []
    
    for i in range(24):  # 24 hours of data
        timestamp = current_time - timedelta(hours=24-i)
        data_point = {
            "timestamp": timestamp.isoformat(),
            "cpu_usage": 0.6 + 0.2 * (i % 12) / 12,  # Cyclical pattern
            "memory_usage": 0.5 + 0.3 * (i % 8) / 8,  # Different cycle
            "gpu_usage": 0.4 + 0.4 * (i % 6) / 6,     # GPU pattern
            "job_queue_length": max(1, 10 - (i % 10)),
            "active_jobs": 5 + (i % 15),
            "cluster_size": 100
        }
        historical_data.append(data_point)
    
    # Create prediction request
    prediction_request = {
        "historical_data": historical_data,
        "prediction_horizon": 12,  # 12 hours ahead
        "confidence_level": 0.95
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict", 
            json=prediction_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            predictions = response.json()
            print("   ✅ Prediction request successful!")
            print(f"   Model Accuracy: {predictions['model_accuracy']}")
            print(f"   Generated at: {predictions['generated_at']}")
            print(f"   Number of predictions: {len(predictions['predictions'])}")
            
            # Show first few predictions
            print("\n   📊 Sample Predictions:")
            for i, pred in enumerate(predictions['predictions'][:3]):
                timestamp = pred['timestamp']
                cpu = pred['predicted_cpu_usage']
                memory = pred['predicted_memory_usage']
                queue = pred['predicted_job_queue_length']
                print(f"     Hour {i+1}: CPU={cpu:.2f}, Memory={memory:.2f}, Queue={queue}")
            
            if len(predictions['predictions']) > 3:
                print(f"     ... and {len(predictions['predictions']) - 3} more predictions")
                
        else:
            print(f"   ❌ Prediction failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Prediction test failed: {e}")
    
    print("\n3. Testing Interactive API Documentation...")
    print(f"   📖 API Docs available at: {base_url}/docs")
    print(f"   📋 OpenAPI Schema: {base_url}/openapi.json")
    
    print("\n🎉 Demo Test Complete!")
    print("\nTo explore the API interactively:")
    print(f"   • Open {base_url}/docs in your browser")
    print("   • Try different prediction parameters")
    print("   • View the automatic API documentation")

if __name__ == "__main__":
    test_workload_predictor()
