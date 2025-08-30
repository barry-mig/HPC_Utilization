# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict, Optional, Union
# import numpy as np
# from datetime import datetime, timedelta
# import logging
# from prometheus_client import Counter, Histogram, generate_latest
# import os
# from enum import Enum
# import heapq
# from dataclasses import dataclass

# DEMO PROJECT - HPC Job Scheduler (Most code commented for presentation)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import numpy as np
from datetime import datetime, timedelta
import logging
from prometheus_client import Counter, Histogram, generate_latest
import os
from enum import Enum
import heapq
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
SCHEDULING_REQUESTS = Counter('scheduling_requests_total', 'Total scheduling requests')
SCHEDULING_LATENCY = Histogram('scheduling_duration_seconds', 'Scheduling request duration')

app = FastAPI(
    title="HPC Scheduler Optimizer",
    description="Advanced job scheduling optimization for HPC clusters",
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

# Enums and Models
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SchedulingPolicy(str, Enum):
    FIFO = "fifo"  # First In, First Out
    SJF = "sjf"   # Shortest Job First
    PRIORITY = "priority"  # Priority-based
    FAIR_SHARE = "fair_share"  # Fair share scheduling
    BACKFILL = "backfill"  # Backfill scheduling

class Job(BaseModel):
    job_id: str
    user_id: str
    priority: int  # 1-10, 10 being highest
    estimated_runtime_minutes: int
    cpu_cores_required: int
    memory_gb_required: float
    gpu_count_required: int = 0
    dependencies: List[str] = []  # Job IDs this job depends on
    submission_time: datetime
    earliest_start_time: Optional[datetime] = None
    deadline: Optional[datetime] = None
    job_type: str = "compute"  # compute, io, memory, gpu
    user_group: str = "default"

class NodeResource(BaseModel):
    node_id: str
    total_cpu_cores: int
    total_memory_gb: float
    total_gpu_count: int
    available_cpu_cores: int
    available_memory_gb: float
    available_gpu_count: int
    is_available: bool = True
    maintenance_until: Optional[datetime] = None

class SchedulingRequest(BaseModel):
    jobs: List[Job]
    cluster_nodes: List[NodeResource]
    policy: SchedulingPolicy = SchedulingPolicy.FAIR_SHARE
    max_schedule_horizon_hours: int = 24
    user_quotas: Optional[Dict[str, Dict[str, Union[int, float]]]] = None

class ScheduledJob(BaseModel):
    job_id: str
    node_id: str
    scheduled_start_time: datetime
    estimated_completion_time: datetime
    allocated_cpu_cores: int
    allocated_memory_gb: float
    allocated_gpu_count: int
    queue_position: int

class SchedulingResult(BaseModel):
    scheduled_jobs: List[ScheduledJob]
    unscheduled_jobs: List[str]  # Job IDs that couldn't be scheduled
    total_jobs: int
    scheduled_count: int
    average_wait_time_minutes: float
    cluster_utilization: Dict[str, float]
    scheduling_efficiency: float
    policy_used: SchedulingPolicy
    generated_at: datetime

@dataclass
class SchedulingEvent:
    time: datetime
    event_type: str  # "start", "end"
    job_id: str
    node_id: str
    resources: Dict[str, Union[int, float]]

class JobScheduler:
    """Advanced HPC job scheduler with multiple scheduling policies"""
    
    def __init__(self):
        self.scheduling_policies = {
            SchedulingPolicy.FIFO: self._schedule_fifo,
            SchedulingPolicy.SJF: self._schedule_sjf,
            SchedulingPolicy.PRIORITY: self._schedule_priority,
            SchedulingPolicy.FAIR_SHARE: self._schedule_fair_share,
            SchedulingPolicy.BACKFILL: self._schedule_backfill
        }
    
    def schedule(self, request: SchedulingRequest) -> SchedulingResult:
        """Main scheduling entry point"""
        start_time = datetime.now()
        
        # Validate dependencies
        self._validate_dependencies(request.jobs)
        
        # Choose and apply scheduling policy
        scheduler_func = self.scheduling_policies.get(request.policy, self._schedule_fair_share)
        result = scheduler_func(request)
        
        # Calculate metrics
        result.scheduling_efficiency = self._calculate_efficiency(result, request)
        result.cluster_utilization = self._calculate_cluster_utilization(result, request)
        result.average_wait_time_minutes = self._calculate_average_wait_time(result, request)
        result.generated_at = datetime.now()
        
        logger.info(f"Scheduled {result.scheduled_count}/{result.total_jobs} jobs using {request.policy}")
        return result
    
    def _schedule_fifo(self, request: SchedulingRequest) -> SchedulingResult:
        """First In, First Out scheduling"""
        # Sort jobs by submission time
        sorted_jobs = sorted(request.jobs, key=lambda j: j.submission_time)
        
        scheduled_jobs = []
        unscheduled_jobs = []
        node_state = self._initialize_node_state(request.cluster_nodes)
        current_time = datetime.now()
        
        for job in sorted_jobs:
            best_node = self._find_available_node(job, node_state, current_time)
            if best_node:
                scheduled_job = self._allocate_job_to_node(job, best_node, current_time, node_state)
                scheduled_jobs.append(scheduled_job)
                self._update_node_state(best_node, job, scheduled_job.estimated_completion_time, node_state)
            else:
                unscheduled_jobs.append(job.job_id)
        
        return SchedulingResult(
            scheduled_jobs=scheduled_jobs,
            unscheduled_jobs=unscheduled_jobs,
            total_jobs=len(request.jobs),
            scheduled_count=len(scheduled_jobs),
            average_wait_time_minutes=0,  # Will be calculated later
            cluster_utilization={},  # Will be calculated later
            scheduling_efficiency=0,  # Will be calculated later
            policy_used=request.policy
        )
    
    def _schedule_sjf(self, request: SchedulingRequest) -> SchedulingResult:
        """Shortest Job First scheduling"""
        # Sort jobs by estimated runtime
        sorted_jobs = sorted(request.jobs, key=lambda j: j.estimated_runtime_minutes)
        
        # Use similar logic to FIFO but with different sorting
        modified_request = SchedulingRequest(
            jobs=sorted_jobs,
            cluster_nodes=request.cluster_nodes,
            policy=request.policy,
            max_schedule_horizon_hours=request.max_schedule_horizon_hours,
            user_quotas=request.user_quotas
        )
        
        return self._schedule_fifo(modified_request)
    
    def _schedule_priority(self, request: SchedulingRequest) -> SchedulingResult:
        """Priority-based scheduling"""
        # Sort jobs by priority (highest first), then by submission time
        sorted_jobs = sorted(
            request.jobs, 
            key=lambda j: (-j.priority, j.submission_time)
        )
        
        modified_request = SchedulingRequest(
            jobs=sorted_jobs,
            cluster_nodes=request.cluster_nodes,
            policy=request.policy,
            max_schedule_horizon_hours=request.max_schedule_horizon_hours,
            user_quotas=request.user_quotas
        )
        
        return self._schedule_fifo(modified_request)
    
    def _schedule_fair_share(self, request: SchedulingRequest) -> SchedulingResult:
        """Fair share scheduling with user quotas"""
        scheduled_jobs = []
        unscheduled_jobs = []
        node_state = self._initialize_node_state(request.cluster_nodes)
        current_time = datetime.now()
        
        # Track resource usage per user
        user_usage = {}
        user_quotas = request.user_quotas or {}
        
        # Calculate fair share weights
        job_queue = self._calculate_fair_share_order(request.jobs, user_quotas)
        
        for job in job_queue:
            # Check user quotas
            if self._check_user_quota(job, user_usage, user_quotas):
                best_node = self._find_available_node(job, node_state, current_time)
                if best_node:
                    scheduled_job = self._allocate_job_to_node(job, best_node, current_time, node_state)
                    scheduled_jobs.append(scheduled_job)
                    self._update_node_state(best_node, job, scheduled_job.estimated_completion_time, node_state)
                    self._update_user_usage(job, user_usage)
                else:
                    unscheduled_jobs.append(job.job_id)
            else:
                unscheduled_jobs.append(job.job_id)
        
        return SchedulingResult(
            scheduled_jobs=scheduled_jobs,
            unscheduled_jobs=unscheduled_jobs,
            total_jobs=len(request.jobs),
            scheduled_count=len(scheduled_jobs),
            average_wait_time_minutes=0,
            cluster_utilization={},
            scheduling_efficiency=0,
            policy_used=request.policy
        )
    
    def _schedule_backfill(self, request: SchedulingRequest) -> SchedulingResult:
        """Backfill scheduling - schedule smaller jobs in gaps"""
        # First, schedule high-priority/large jobs
        priority_jobs = [j for j in request.jobs if j.priority >= 7]
        regular_jobs = [j for j in request.jobs if j.priority < 7]
        
        # Sort priority jobs by priority, regular jobs by size (for backfill)
        priority_jobs.sort(key=lambda j: -j.priority)
        regular_jobs.sort(key=lambda j: j.estimated_runtime_minutes)
        
        scheduled_jobs = []
        unscheduled_jobs = []
        node_state = self._initialize_node_state(request.cluster_nodes)
        current_time = datetime.now()
        
        # Schedule priority jobs first
        for job in priority_jobs:
            best_node = self._find_available_node(job, node_state, current_time)
            if best_node:
                scheduled_job = self._allocate_job_to_node(job, best_node, current_time, node_state)
                scheduled_jobs.append(scheduled_job)
                self._update_node_state(best_node, job, scheduled_job.estimated_completion_time, node_state)
            else:
                unscheduled_jobs.append(job.job_id)
        
        # Backfill with smaller jobs
        for job in regular_jobs:
            best_node = self._find_available_node(job, node_state, current_time)
            if best_node:
                scheduled_job = self._allocate_job_to_node(job, best_node, current_time, node_state)
                scheduled_jobs.append(scheduled_job)
                self._update_node_state(best_node, job, scheduled_job.estimated_completion_time, node_state)
            else:
                unscheduled_jobs.append(job.job_id)
        
        return SchedulingResult(
            scheduled_jobs=scheduled_jobs,
            unscheduled_jobs=unscheduled_jobs,
            total_jobs=len(request.jobs),
            scheduled_count=len(scheduled_jobs),
            average_wait_time_minutes=0,
            cluster_utilization={},
            scheduling_efficiency=0,
            policy_used=request.policy
        )
    
    def _initialize_node_state(self, nodes: List[NodeResource]) -> Dict:
        """Initialize node state tracking"""
        return {
            node.node_id: {
                'available_cpu': node.available_cpu_cores,
                'available_memory': node.available_memory_gb,
                'available_gpu': node.available_gpu_count,
                'is_available': node.is_available,
                'scheduled_jobs': [],
                'next_available_time': datetime.now()
            }
            for node in nodes
        }
    
    def _find_available_node(self, job: Job, node_state: Dict, current_time: datetime) -> Optional[str]:
        """Find the best available node for a job"""
        best_node = None
        best_score = -1
        
        for node_id, state in node_state.items():
            if (state['is_available'] and 
                state['available_cpu'] >= job.cpu_cores_required and
                state['available_memory'] >= job.memory_gb_required and
                state['available_gpu'] >= job.gpu_count_required):
                
                # Calculate scoring based on resource utilization and availability
                cpu_util = job.cpu_cores_required / state['available_cpu']
                memory_util = job.memory_gb_required / state['available_memory']
                
                # Prefer balanced utilization
                balance_score = 1.0 - abs(cpu_util - memory_util)
                availability_score = 1.0 if state['next_available_time'] <= current_time else 0.5
                
                score = balance_score * 0.6 + availability_score * 0.4
                
                if score > best_score:
                    best_score = score
                    best_node = node_id
        
        return best_node
    
    def _allocate_job_to_node(self, job: Job, node_id: str, current_time: datetime, 
                            node_state: Dict) -> ScheduledJob:
        """Allocate a job to a specific node"""
        start_time = max(current_time, job.earliest_start_time or current_time)
        start_time = max(start_time, node_state[node_id]['next_available_time'])
        
        completion_time = start_time + timedelta(minutes=job.estimated_runtime_minutes)
        
        return ScheduledJob(
            job_id=job.job_id,
            node_id=node_id,
            scheduled_start_time=start_time,
            estimated_completion_time=completion_time,
            allocated_cpu_cores=job.cpu_cores_required,
            allocated_memory_gb=job.memory_gb_required,
            allocated_gpu_count=job.gpu_count_required,
            queue_position=len(node_state[node_id]['scheduled_jobs'])
        )
    
    def _update_node_state(self, node_id: str, job: Job, completion_time: datetime, node_state: Dict):
        """Update node state after job allocation"""
        state = node_state[node_id]
        state['available_cpu'] -= job.cpu_cores_required
        state['available_memory'] -= job.memory_gb_required
        state['available_gpu'] -= job.gpu_count_required
        state['scheduled_jobs'].append(job.job_id)
        state['next_available_time'] = completion_time
    
    def _calculate_fair_share_order(self, jobs: List[Job], user_quotas: Dict) -> List[Job]:
        """Calculate fair share order for job scheduling"""
        # Simplified fair share - in practice this would be more complex
        user_job_counts = {}
        for job in jobs:
            user_job_counts[job.user_id] = user_job_counts.get(job.user_id, 0) + 1
        
        # Sort jobs to balance between users
        def fair_share_key(job):
            user_priority = user_quotas.get(job.user_id, {}).get('priority', 1)
            user_job_count = user_job_counts[job.user_id]
            return (-job.priority * user_priority / user_job_count, job.submission_time)
        
        return sorted(jobs, key=fair_share_key)
    
    def _check_user_quota(self, job: Job, user_usage: Dict, user_quotas: Dict) -> bool:
        """Check if user is within quota limits"""
        if not user_quotas or job.user_id not in user_quotas:
            return True
        
        user_quota = user_quotas[job.user_id]
        current_usage = user_usage.get(job.user_id, {'cpu': 0, 'memory': 0, 'jobs': 0})
        
        max_cpu = user_quota.get('max_cpu_cores', float('inf'))
        max_memory = user_quota.get('max_memory_gb', float('inf'))
        max_jobs = user_quota.get('max_concurrent_jobs', float('inf'))
        
        return (current_usage['cpu'] + job.cpu_cores_required <= max_cpu and
                current_usage['memory'] + job.memory_gb_required <= max_memory and
                current_usage['jobs'] + 1 <= max_jobs)
    
    def _update_user_usage(self, job: Job, user_usage: Dict):
        """Update user resource usage tracking"""
        if job.user_id not in user_usage:
            user_usage[job.user_id] = {'cpu': 0, 'memory': 0, 'jobs': 0}
        
        user_usage[job.user_id]['cpu'] += job.cpu_cores_required
        user_usage[job.user_id]['memory'] += job.memory_gb_required
        user_usage[job.user_id]['jobs'] += 1
    
    def _validate_dependencies(self, jobs: List[Job]):
        """Validate job dependencies"""
        job_ids = {job.job_id for job in jobs}
        for job in jobs:
            for dep in job.dependencies:
                if dep not in job_ids:
                    raise ValueError(f"Job {job.job_id} depends on non-existent job {dep}")
    
    def _calculate_efficiency(self, result: SchedulingResult, request: SchedulingRequest) -> float:
        """Calculate scheduling efficiency"""
        if result.total_jobs == 0:
            return 0.0
        return result.scheduled_count / result.total_jobs
    
    def _calculate_cluster_utilization(self, result: SchedulingResult, 
                                     request: SchedulingRequest) -> Dict[str, float]:
        """Calculate cluster resource utilization"""
        total_cpu = sum(node.total_cpu_cores for node in request.cluster_nodes)
        total_memory = sum(node.total_memory_gb for node in request.cluster_nodes)
        total_gpu = sum(node.total_gpu_count for node in request.cluster_nodes)
        
        used_cpu = sum(job.allocated_cpu_cores for job in result.scheduled_jobs)
        used_memory = sum(job.allocated_memory_gb for job in result.scheduled_jobs)
        used_gpu = sum(job.allocated_gpu_count for job in result.scheduled_jobs)
        
        return {
            'cpu': used_cpu / total_cpu if total_cpu > 0 else 0,
            'memory': used_memory / total_memory if total_memory > 0 else 0,
            'gpu': used_gpu / total_gpu if total_gpu > 0 else 0,
        }
    
    def _calculate_average_wait_time(self, result: SchedulingResult, 
                                   request: SchedulingRequest) -> float:
        """Calculate average wait time for scheduled jobs"""
        if not result.scheduled_jobs:
            return 0.0
        
        job_dict = {job.job_id: job for job in request.jobs}
        total_wait_time = 0
        
        for scheduled_job in result.scheduled_jobs:
            original_job = job_dict[scheduled_job.job_id]
            wait_time = (scheduled_job.scheduled_start_time - original_job.submission_time).total_seconds() / 60
            total_wait_time += max(0, wait_time)
        
        return total_wait_time / len(result.scheduled_jobs)

# Global scheduler instance
scheduler = JobScheduler()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "scheduler-optimizer",
        "available_policies": [policy.value for policy in SchedulingPolicy]
    }

@app.post("/schedule", response_model=SchedulingResult)
async def schedule_jobs(request: SchedulingRequest):
    """
    Schedule HPC jobs using the specified policy
    """
    SCHEDULING_REQUESTS.inc()
    
    with SCHEDULING_LATENCY.time():
        try:
            result = scheduler.schedule(request)
            logger.info(f"Scheduled {result.scheduled_count}/{result.total_jobs} jobs")
            return result
        except Exception as e:
            logger.error(f"Scheduling error: {e}")
            raise HTTPException(status_code=500, detail=f"Scheduling failed: {str(e)}")

@app.get("/policies")
async def get_scheduling_policies():
    """Get available scheduling policies"""
    return {
        "policies": [
            {
                "name": policy.value,
                "description": {
                    "fifo": "First In, First Out - Jobs are scheduled in submission order",
                    "sjf": "Shortest Job First - Shorter jobs are prioritized",
                    "priority": "Priority-based - Jobs with higher priority are scheduled first",
                    "fair_share": "Fair Share - Resources are distributed fairly among users",
                    "backfill": "Backfill - Small jobs fill gaps left by larger jobs"
                }[policy.value]
            }
            for policy in SchedulingPolicy
        ],
        "default": "fair_share"
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/scheduler/stats")
async def get_scheduler_stats():
    """Get scheduler statistics"""
    return {
        "total_requests": SCHEDULING_REQUESTS._value._value,
        "average_latency_seconds": 0.05,  # Placeholder
        "supported_policies": len(SchedulingPolicy),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8002)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
