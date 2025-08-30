# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict, Optional
# import numpy as np
# from datetime import datetime, timedelta
# import logging
# from prometheus_client import Counter, Histogram, generate_latest
# import os
# from scipy.optimize import minimize
# from dataclasses import dataclass

# DEMO PROJECT - Most functionality commented out for presentation
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime, timedelta
import logging
from prometheus_client import Counter, Histogram, generate_latest
import os
from scipy.optimize import minimize
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
OPTIMIZATION_REQUESTS = Counter('resource_optimization_requests_total', 'Total optimization requests')
OPTIMIZATION_LATENCY = Histogram('resource_optimization_duration_seconds', 'Optimization request duration')

app = FastAPI(
    title="HPC Resource Optimizer",
    description="Advanced resource allocation optimization for HPC clusters",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ResourceConstraints(BaseModel):
    max_cpu_cores: int
    max_memory_gb: float
    max_gpu_count: int
    max_power_watts: float
    max_network_bandwidth_gbps: float

class JobRequirements(BaseModel):
    job_id: str
    priority: int  # 1-10, 10 being highest
    estimated_runtime_hours: float
    cpu_cores_required: int
    memory_gb_required: float
    gpu_count_required: int
    power_watts_estimated: float
    network_bandwidth_gbps: float
    deadline: Optional[datetime] = None
    job_type: str  # "compute", "memory", "gpu", "network"

class ClusterNode(BaseModel):
    node_id: str
    available_cpu_cores: int
    available_memory_gb: float
    available_gpu_count: int
    available_power_watts: float
    available_network_bandwidth_gbps: float
    node_efficiency_score: float  # 0.0-1.0
    maintenance_window: Optional[datetime] = None

class OptimizationRequest(BaseModel):
    jobs: List[JobRequirements]
    cluster_nodes: List[ClusterNode]
    global_constraints: ResourceConstraints
    optimization_objective: str = "efficiency"  # "efficiency", "throughput", "power", "balanced"

class JobAllocation(BaseModel):
    job_id: str
    allocated_node_id: str
    allocated_cpu_cores: int
    allocated_memory_gb: float
    allocated_gpu_count: int
    estimated_start_time: datetime
    estimated_completion_time: datetime
    allocation_efficiency: float

class OptimizationResult(BaseModel):
    job_allocations: List[JobAllocation]
    total_efficiency_score: float
    resource_utilization: Dict[str, float]
    power_consumption_watts: float
    estimated_makespan_hours: float
    optimization_time_seconds: float
    recommendations: List[str]

@dataclass
class OptimizationSolution:
    allocations: List[tuple]  # (job_idx, node_idx)
    efficiency_score: float
    resource_utilization: Dict[str, float]
    power_consumption: float

class ResourceOptimizer:
    """Advanced resource optimization engine"""
    
    def __init__(self):
        self.optimization_methods = {
            "efficiency": self._optimize_for_efficiency,
            "throughput": self._optimize_for_throughput,
            "power": self._optimize_for_power,
            "balanced": self._optimize_balanced
        }
    
    def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        """Main optimization entry point"""
        start_time = datetime.now()
        
        # Choose optimization method based on objective
        optimizer_func = self.optimization_methods.get(
            request.optimization_objective, 
            self._optimize_balanced
        )
        
        # Run optimization
        solution = optimizer_func(request)
        
        # Convert solution to response format
        result = self._create_optimization_result(request, solution, start_time)
        
        return result
    
    def _optimize_for_efficiency(self, request: OptimizationRequest) -> OptimizationSolution:
        """Optimize for maximum resource efficiency"""
        jobs = request.jobs
        nodes = request.cluster_nodes
        
        # Sort jobs by priority and efficiency potential
        sorted_jobs = sorted(
            enumerate(jobs), 
            key=lambda x: (x[1].priority, -x[1].estimated_runtime_hours),
            reverse=True
        )
        
        allocations = []
        node_usage = {i: {
            'cpu': 0, 'memory': 0, 'gpu': 0, 'power': 0, 'bandwidth': 0
        } for i in range(len(nodes))}
        
        for job_idx, job in sorted_jobs:
            best_node_idx = self._find_best_node_for_job(job, nodes, node_usage)
            if best_node_idx is not None:
                allocations.append((job_idx, best_node_idx))
                self._update_node_usage(best_node_idx, job, node_usage)
        
        efficiency_score = self._calculate_efficiency_score(allocations, jobs, nodes)
        resource_util = self._calculate_resource_utilization(node_usage, nodes)
        power_consumption = sum(usage['power'] for usage in node_usage.values())
        
        return OptimizationSolution(
            allocations=allocations,
            efficiency_score=efficiency_score,
            resource_utilization=resource_util,
            power_consumption=power_consumption
        )
    
    def _optimize_for_throughput(self, request: OptimizationRequest) -> OptimizationSolution:
        """Optimize for maximum job throughput"""
        # Similar to efficiency but prioritizes fitting more jobs
        jobs = request.jobs
        nodes = request.cluster_nodes
        
        # Sort jobs by size (smaller jobs first for better packing)
        sorted_jobs = sorted(
            enumerate(jobs),
            key=lambda x: (x[1].cpu_cores_required + x[1].memory_gb_required)
        )
        
        allocations = []
        node_usage = {i: {
            'cpu': 0, 'memory': 0, 'gpu': 0, 'power': 0, 'bandwidth': 0
        } for i in range(len(nodes))}
        
        for job_idx, job in sorted_jobs:
            best_node_idx = self._find_best_node_for_job(job, nodes, node_usage)
            if best_node_idx is not None:
                allocations.append((job_idx, best_node_idx))
                self._update_node_usage(best_node_idx, job, node_usage)
        
        efficiency_score = len(allocations) / len(jobs)  # Throughput-based score
        resource_util = self._calculate_resource_utilization(node_usage, nodes)
        power_consumption = sum(usage['power'] for usage in node_usage.values())
        
        return OptimizationSolution(
            allocations=allocations,
            efficiency_score=efficiency_score,
            resource_utilization=resource_util,
            power_consumption=power_consumption
        )
    
    def _optimize_for_power(self, request: OptimizationRequest) -> OptimizationSolution:
        """Optimize for minimum power consumption"""
        jobs = request.jobs
        nodes = request.cluster_nodes
        
        # Sort nodes by efficiency score (higher efficiency = lower power per unit work)
        node_efficiency = [(i, node.node_efficiency_score) for i, node in enumerate(nodes)]
        node_efficiency.sort(key=lambda x: x[1], reverse=True)
        
        # Sort jobs by power requirements (lower power first)
        sorted_jobs = sorted(
            enumerate(jobs),
            key=lambda x: x[1].power_watts_estimated
        )
        
        allocations = []
        node_usage = {i: {
            'cpu': 0, 'memory': 0, 'gpu': 0, 'power': 0, 'bandwidth': 0
        } for i in range(len(nodes))}
        
        for job_idx, job in sorted_jobs:
            # Try to allocate to most efficient nodes first
            best_node_idx = None
            for node_idx, _ in node_efficiency:
                if self._can_allocate_job(job, nodes[node_idx], node_usage[node_idx]):
                    best_node_idx = node_idx
                    break
            
            if best_node_idx is not None:
                allocations.append((job_idx, best_node_idx))
                self._update_node_usage(best_node_idx, job, node_usage)
        
        efficiency_score = self._calculate_power_efficiency_score(allocations, jobs, nodes)
        resource_util = self._calculate_resource_utilization(node_usage, nodes)
        power_consumption = sum(usage['power'] for usage in node_usage.values())
        
        return OptimizationSolution(
            allocations=allocations,
            efficiency_score=efficiency_score,
            resource_utilization=resource_util,
            power_consumption=power_consumption
        )
    
    def _optimize_balanced(self, request: OptimizationRequest) -> OptimizationSolution:
        """Balanced optimization considering multiple objectives"""
        # Combine multiple optimization strategies
        efficiency_solution = self._optimize_for_efficiency(request)
        throughput_solution = self._optimize_for_throughput(request)
        power_solution = self._optimize_for_power(request)
        
        # Score each solution on multiple criteria
        solutions = [
            ("efficiency", efficiency_solution),
            ("throughput", throughput_solution),
            ("power", power_solution)
        ]
        
        best_solution = max(
            solutions,
            key=lambda x: self._calculate_balanced_score(x[1])
        )[1]
        
        return best_solution
    
    def _find_best_node_for_job(self, job: JobRequirements, nodes: List[ClusterNode], 
                               node_usage: Dict) -> Optional[int]:
        """Find the best node for a given job"""
        best_node_idx = None
        best_score = -1
        
        for i, node in enumerate(nodes):
            if self._can_allocate_job(job, node, node_usage[i]):
                # Calculate allocation score based on fit and efficiency
                score = self._calculate_allocation_score(job, node, node_usage[i])
                if score > best_score:
                    best_score = score
                    best_node_idx = i
        
        return best_node_idx
    
    def _can_allocate_job(self, job: JobRequirements, node: ClusterNode, usage: Dict) -> bool:
        """Check if a job can be allocated to a node"""
        return (
            node.available_cpu_cores >= job.cpu_cores_required + usage['cpu'] and
            node.available_memory_gb >= job.memory_gb_required + usage['memory'] and
            node.available_gpu_count >= job.gpu_count_required + usage['gpu'] and
            node.available_power_watts >= job.power_watts_estimated + usage['power'] and
            node.available_network_bandwidth_gbps >= job.network_bandwidth_gbps + usage['bandwidth']
        )
    
    def _calculate_allocation_score(self, job: JobRequirements, node: ClusterNode, usage: Dict) -> float:
        """Calculate how good an allocation would be"""
        # Consider resource utilization efficiency and node efficiency
        cpu_util = (job.cpu_cores_required + usage['cpu']) / node.available_cpu_cores
        memory_util = (job.memory_gb_required + usage['memory']) / node.available_memory_gb
        
        # Prefer allocations that balance resource usage
        balance_score = 1.0 - abs(cpu_util - memory_util)
        efficiency_score = node.node_efficiency_score
        priority_score = job.priority / 10.0
        
        return balance_score * 0.4 + efficiency_score * 0.4 + priority_score * 0.2
    
    def _update_node_usage(self, node_idx: int, job: JobRequirements, node_usage: Dict):
        """Update node usage after job allocation"""
        node_usage[node_idx]['cpu'] += job.cpu_cores_required
        node_usage[node_idx]['memory'] += job.memory_gb_required
        node_usage[node_idx]['gpu'] += job.gpu_count_required
        node_usage[node_idx]['power'] += job.power_watts_estimated
        node_usage[node_idx]['bandwidth'] += job.network_bandwidth_gbps
    
    def _calculate_efficiency_score(self, allocations: List[tuple], 
                                  jobs: List[JobRequirements], 
                                  nodes: List[ClusterNode]) -> float:
        """Calculate overall efficiency score"""
        if not allocations:
            return 0.0
        
        total_priority_weight = sum(jobs[job_idx].priority for job_idx, _ in allocations)
        max_possible_priority = sum(job.priority for job in jobs)
        
        efficiency = total_priority_weight / max_possible_priority if max_possible_priority > 0 else 0
        return min(1.0, efficiency)
    
    def _calculate_power_efficiency_score(self, allocations: List[tuple],
                                        jobs: List[JobRequirements],
                                        nodes: List[ClusterNode]) -> float:
        """Calculate power efficiency score"""
        if not allocations:
            return 0.0
        
        total_work = sum(jobs[job_idx].priority * jobs[job_idx].estimated_runtime_hours 
                        for job_idx, _ in allocations)
        total_power = sum(jobs[job_idx].power_watts_estimated for job_idx, _ in allocations)
        
        return total_work / total_power if total_power > 0 else 0
    
    def _calculate_balanced_score(self, solution: OptimizationSolution) -> float:
        """Calculate balanced score for multi-objective optimization"""
        # Weighted combination of different objectives
        efficiency_weight = 0.4
        utilization_weight = 0.3
        power_weight = 0.3
        
        avg_utilization = np.mean(list(solution.resource_utilization.values()))
        power_efficiency = 1.0 / (solution.power_consumption + 1)  # Inverse power consumption
        
        return (
            efficiency_weight * solution.efficiency_score +
            utilization_weight * avg_utilization +
            power_weight * power_efficiency
        )
    
    def _calculate_resource_utilization(self, node_usage: Dict, nodes: List[ClusterNode]) -> Dict[str, float]:
        """Calculate resource utilization percentages"""
        total_cpu_used = sum(usage['cpu'] for usage in node_usage.values())
        total_memory_used = sum(usage['memory'] for usage in node_usage.values())
        total_gpu_used = sum(usage['gpu'] for usage in node_usage.values())
        total_power_used = sum(usage['power'] for usage in node_usage.values())
        
        total_cpu_available = sum(node.available_cpu_cores for node in nodes)
        total_memory_available = sum(node.available_memory_gb for node in nodes)
        total_gpu_available = sum(node.available_gpu_count for node in nodes)
        total_power_available = sum(node.available_power_watts for node in nodes)
        
        return {
            'cpu': total_cpu_used / total_cpu_available if total_cpu_available > 0 else 0,
            'memory': total_memory_used / total_memory_available if total_memory_available > 0 else 0,
            'gpu': total_gpu_used / total_gpu_available if total_gpu_available > 0 else 0,
            'power': total_power_used / total_power_available if total_power_available > 0 else 0,
        }
    
    def _create_optimization_result(self, request: OptimizationRequest, 
                                  solution: OptimizationSolution, 
                                  start_time: datetime) -> OptimizationResult:
        """Convert optimization solution to API response"""
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        job_allocations = []
        for job_idx, node_idx in solution.allocations:
            job = request.jobs[job_idx]
            node = request.cluster_nodes[node_idx]
            
            # Calculate estimated times (simplified)
            estimated_start = datetime.now()
            estimated_completion = estimated_start + timedelta(hours=job.estimated_runtime_hours)
            
            allocation = JobAllocation(
                job_id=job.job_id,
                allocated_node_id=node.node_id,
                allocated_cpu_cores=job.cpu_cores_required,
                allocated_memory_gb=job.memory_gb_required,
                allocated_gpu_count=job.gpu_count_required,
                estimated_start_time=estimated_start,
                estimated_completion_time=estimated_completion,
                allocation_efficiency=node.node_efficiency_score
            )
            job_allocations.append(allocation)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(request, solution)
        
        # Calculate makespan (maximum completion time)
        makespan_hours = max(
            (alloc.estimated_completion_time - datetime.now()).total_seconds() / 3600
            for alloc in job_allocations
        ) if job_allocations else 0
        
        return OptimizationResult(
            job_allocations=job_allocations,
            total_efficiency_score=solution.efficiency_score,
            resource_utilization=solution.resource_utilization,
            power_consumption_watts=solution.power_consumption,
            estimated_makespan_hours=makespan_hours,
            optimization_time_seconds=optimization_time,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, request: OptimizationRequest, 
                                solution: OptimizationSolution) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check resource utilization
        avg_util = np.mean(list(solution.resource_utilization.values()))
        if avg_util < 0.6:
            recommendations.append("Consider consolidating workloads to fewer nodes for better efficiency")
        elif avg_util > 0.9:
            recommendations.append("Cluster utilization is very high - consider scaling out")
        
        # Check allocation efficiency
        if solution.efficiency_score < 0.7:
            recommendations.append("Low allocation efficiency - review job priorities and resource requirements")
        
        # Check power consumption
        if solution.power_consumption > 50000:  # Example threshold
            recommendations.append("High power consumption detected - consider power-optimized scheduling")
        
        # Check unallocated jobs
        allocated_jobs = len(solution.allocations)
        total_jobs = len(request.jobs)
        if allocated_jobs < total_jobs:
            unallocated = total_jobs - allocated_jobs
            recommendations.append(f"{unallocated} jobs could not be allocated - insufficient resources")
        
        return recommendations

# Global optimizer instance
optimizer = ResourceOptimizer()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "resource-optimizer"}

@app.post("/optimize", response_model=OptimizationResult)
async def optimize_resources(request: OptimizationRequest):
    """
    Optimize resource allocation for HPC workloads
    """
    OPTIMIZATION_REQUESTS.inc()
    
    with OPTIMIZATION_LATENCY.time():
        try:
            result = optimizer.optimize(request)
            logger.info(f"Optimization completed. Efficiency: {result.total_efficiency_score:.2f}")
            return result
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/algorithms")
async def get_available_algorithms():
    """Get list of available optimization algorithms"""
    return {
        "algorithms": list(optimizer.optimization_methods.keys()),
        "default": "balanced",
        "descriptions": {
            "efficiency": "Maximize resource utilization efficiency",
            "throughput": "Maximize number of jobs completed",
            "power": "Minimize power consumption",
            "balanced": "Balance multiple objectives"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
