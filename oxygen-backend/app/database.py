"""
Supabase database abstraction layer for Oxygen backend.
Replaces in-memory storage with persistent Supabase storage.
"""
import os
import json
import time
from typing import Optional
from supabase import create_client, Client

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://zurxhmogpssqquurvakm.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_secret_2K63EwfcN8saV2ugRYJfyQ_N-muElv3")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============== JOBS ==============

def get_job(job_id: str) -> Optional[dict]:
    """Get a job by ID"""
    try:
        result = supabase.table("jobs").select("*").eq("id", job_id).execute()
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        print(f"Error getting job {job_id}: {e}")
        return None

def create_job(job_data: dict) -> dict:
    """Create a new job"""
    try:
        # Convert nested objects to JSON strings for storage
        db_data = {
            "id": job_data["id"],
            "customer_id": job_data["customer_id"],
            "kernel_type": job_data["kernel_type"],
            "status": job_data["status"],
            "params": json.dumps(job_data.get("params", {})),
            "total_fragments": job_data.get("total_fragments", 0),
            "verified_fragments": job_data.get("verified_fragments", 0),
            "result": json.dumps(job_data.get("result")) if job_data.get("result") else None,
            "created_at": job_data["created_at"],
            "completed_at": job_data.get("completed_at"),
            "lifecycle_events": json.dumps(job_data.get("lifecycle", []))
        }
        result = supabase.table("jobs").insert(db_data).execute()
        return result.data[0] if result.data else job_data
    except Exception as e:
        print(f"Error creating job: {e}")
        return job_data

def update_job(job_id: str, updates: dict) -> Optional[dict]:
    """Update a job"""
    try:
        db_updates = {}
        for key, value in updates.items():
            if key in ["params", "result", "lifecycle_events", "lifecycle"]:
                db_updates[key if key != "lifecycle" else "lifecycle_events"] = json.dumps(value)
            else:
                db_updates[key] = value
        result = supabase.table("jobs").update(db_updates).eq("id", job_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error updating job {job_id}: {e}")
        return None

def list_jobs(customer_id: Optional[str] = None) -> list:
    """List all jobs, optionally filtered by customer_id"""
    try:
        query = supabase.table("jobs").select("*")
        if customer_id:
            query = query.eq("customer_id", customer_id)
        result = query.order("created_at", desc=True).execute()
        # Parse JSON fields
        jobs = []
        for job in result.data:
            job["params"] = json.loads(job["params"]) if job.get("params") else {}
            job["result"] = json.loads(job["result"]) if job.get("result") else None
            job["lifecycle"] = json.loads(job["lifecycle_events"]) if job.get("lifecycle_events") else []
            jobs.append(job)
        return jobs
    except Exception as e:
        print(f"Error listing jobs: {e}")
        return []

# ============== FRAGMENTS ==============

def get_fragment(fragment_id: str) -> Optional[dict]:
    """Get a fragment by ID"""
    try:
        result = supabase.table("fragments").select("*").eq("id", fragment_id).execute()
        if result.data and len(result.data) > 0:
            frag = result.data[0]
            frag["shard_params"] = json.loads(frag["shard_params"]) if frag.get("shard_params") else {}
            frag["result"] = json.loads(frag["result"]) if frag.get("result") else None
            return frag
        return None
    except Exception as e:
        print(f"Error getting fragment {fragment_id}: {e}")
        return None

def create_fragment(fragment_data: dict) -> dict:
    """Create a new fragment"""
    try:
        db_data = {
            "id": fragment_data["id"],
            "job_id": fragment_data["job_id"],
            "fragment_index": fragment_data["fragment_index"],
            "shard_id": fragment_data["shard_id"],
            "shard_params": json.dumps(fragment_data.get("shard_params", {})),
            "status": fragment_data["status"],
            "result": json.dumps(fragment_data.get("result")) if fragment_data.get("result") else None,
            "created_at": fragment_data["created_at"],
            "completed_at": fragment_data.get("completed_at")
        }
        result = supabase.table("fragments").insert(db_data).execute()
        return result.data[0] if result.data else fragment_data
    except Exception as e:
        print(f"Error creating fragment: {e}")
        return fragment_data

def update_fragment(fragment_id: str, updates: dict) -> Optional[dict]:
    """Update a fragment"""
    try:
        db_updates = {}
        for key, value in updates.items():
            if key in ["shard_params", "result"]:
                db_updates[key] = json.dumps(value)
            else:
                db_updates[key] = value
        result = supabase.table("fragments").update(db_updates).eq("id", fragment_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error updating fragment {fragment_id}: {e}")
        return None

def list_fragments_by_job(job_id: str) -> list:
    """List all fragments for a job"""
    try:
        result = supabase.table("fragments").select("*").eq("job_id", job_id).order("fragment_index").execute()
        fragments = []
        for frag in result.data:
            frag["shard_params"] = json.loads(frag["shard_params"]) if frag.get("shard_params") else {}
            frag["result"] = json.loads(frag["result"]) if frag.get("result") else None
            fragments.append(frag)
        return fragments
    except Exception as e:
        print(f"Error listing fragments for job {job_id}: {e}")
        return []

# ============== TASKS ==============

def get_task(task_id: str) -> Optional[dict]:
    """Get a task by ID"""
    try:
        result = supabase.table("tasks").select("*").eq("id", task_id).execute()
        if result.data and len(result.data) > 0:
            task = result.data[0]
            task["result"] = json.loads(task["result"]) if task.get("result") else None
            return task
        return None
    except Exception as e:
        print(f"Error getting task {task_id}: {e}")
        return None

def create_task(task_data: dict) -> dict:
    """Create a new task"""
    try:
        db_data = {
            "id": task_data["id"],
            "fragment_id": task_data["fragment_id"],
            "worker_id": task_data.get("worker_id"),
            "status": task_data["status"],
            "result": json.dumps(task_data.get("result")) if task_data.get("result") else None,
            "created_at": task_data["created_at"],
            "assigned_at": task_data.get("assigned_at"),
            "completed_at": task_data.get("completed_at"),
            "lease_expires_at": task_data.get("lease_expires_at")
        }
        result = supabase.table("tasks").insert(db_data).execute()
        return result.data[0] if result.data else task_data
    except Exception as e:
        print(f"Error creating task: {e}")
        return task_data

def update_task(task_id: str, updates: dict) -> Optional[dict]:
    """Update a task"""
    try:
        db_updates = {}
        for key, value in updates.items():
            if key == "result":
                db_updates[key] = json.dumps(value)
            else:
                db_updates[key] = value
        result = supabase.table("tasks").update(db_updates).eq("id", task_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error updating task {task_id}: {e}")
        return None

def list_pending_tasks() -> list:
    """List all pending tasks (task queue)"""
    try:
        result = supabase.table("tasks").select("*").eq("status", "pending").order("created_at").execute()
        tasks = []
        for task in result.data:
            task["result"] = json.loads(task["result"]) if task.get("result") else None
            tasks.append(task)
        return tasks
    except Exception as e:
        print(f"Error listing pending tasks: {e}")
        return []

def list_tasks_by_fragment(fragment_id: str) -> list:
    """List all tasks for a fragment"""
    try:
        result = supabase.table("tasks").select("*").eq("fragment_id", fragment_id).execute()
        tasks = []
        for task in result.data:
            task["result"] = json.loads(task["result"]) if task.get("result") else None
            tasks.append(task)
        return tasks
    except Exception as e:
        print(f"Error listing tasks for fragment {fragment_id}: {e}")
        return []

# ============== WORKERS ==============

def get_worker(worker_id: str) -> Optional[dict]:
    """Get a worker by ID"""
    try:
        result = supabase.table("workers").select("*").eq("id", worker_id).execute()
        if result.data and len(result.data) > 0:
            worker = result.data[0]
            worker["device_info"] = json.loads(worker["device_info"]) if worker.get("device_info") else {}
            worker["capabilities"] = json.loads(worker["capabilities"]) if worker.get("capabilities") else {}
            return worker
        return None
    except Exception as e:
        print(f"Error getting worker {worker_id}: {e}")
        return None

def create_worker(worker_data: dict) -> dict:
    """Create a new worker"""
    try:
        db_data = {
            "id": worker_data["id"],
            "name": worker_data["name"],
            "device_info": json.dumps(worker_data.get("device_info", {})),
            "capabilities": json.dumps(worker_data.get("capabilities", {})),
            "current_task_id": worker_data.get("current_task_id"),
            "connected_at": worker_data["connected_at"],
            "last_heartbeat": worker_data["last_heartbeat"],
            "total_tasks_completed": worker_data.get("total_tasks_completed", 0),
            "total_earnings": worker_data.get("total_earnings", 0.0)
        }
        result = supabase.table("workers").insert(db_data).execute()
        return result.data[0] if result.data else worker_data
    except Exception as e:
        print(f"Error creating worker: {e}")
        return worker_data

def update_worker(worker_id: str, updates: dict) -> Optional[dict]:
    """Update a worker"""
    try:
        db_updates = {}
        for key, value in updates.items():
            if key in ["device_info", "capabilities"]:
                db_updates[key] = json.dumps(value)
            else:
                db_updates[key] = value
        result = supabase.table("workers").update(db_updates).eq("id", worker_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error updating worker {worker_id}: {e}")
        return None

def delete_worker(worker_id: str) -> bool:
    """Delete a worker"""
    try:
        supabase.table("workers").delete().eq("id", worker_id).execute()
        return True
    except Exception as e:
        print(f"Error deleting worker {worker_id}: {e}")
        return False

def list_workers() -> list:
    """List all workers"""
    try:
        result = supabase.table("workers").select("*").execute()
        workers = []
        for worker in result.data:
            worker["device_info"] = json.loads(worker["device_info"]) if worker.get("device_info") else {}
            worker["capabilities"] = json.loads(worker["capabilities"]) if worker.get("capabilities") else {}
            workers.append(worker)
        return workers
    except Exception as e:
        print(f"Error listing workers: {e}")
        return []

# ============== DATA STORE ==============

def store_data(data_id: str, data: bytes) -> bool:
    """Store binary data"""
    try:
        import base64
        db_data = {
            "id": data_id,
            "data": base64.b64encode(data).decode('utf-8'),
            "created_at": time.time()
        }
        # Use upsert to handle duplicates
        supabase.table("data_store").upsert(db_data).execute()
        return True
    except Exception as e:
        print(f"Error storing data {data_id}: {e}")
        return False

def get_data(data_id: str) -> Optional[bytes]:
    """Get binary data by ID"""
    try:
        import base64
        result = supabase.table("data_store").select("data").eq("id", data_id).execute()
        if result.data and len(result.data) > 0:
            return base64.b64decode(result.data[0]["data"])
        return None
    except Exception as e:
        print(f"Error getting data {data_id}: {e}")
        return None

# ============== PAYMENTS ==============

def create_payment(payment_data: dict) -> dict:
    """Create a new payment"""
    try:
        db_data = {
            "id": payment_data["id"],
            "job_id": payment_data["job_id"],
            "worker_id": payment_data.get("worker_id"),
            "amount": payment_data["amount"],
            "status": payment_data["status"],
            "created_at": payment_data["created_at"],
            "released_at": payment_data.get("released_at")
        }
        result = supabase.table("payments").insert(db_data).execute()
        return result.data[0] if result.data else payment_data
    except Exception as e:
        print(f"Error creating payment: {e}")
        return payment_data

def list_payments_by_job(job_id: str) -> list:
    """List all payments for a job"""
    try:
        result = supabase.table("payments").select("*").eq("job_id", job_id).execute()
        return result.data
    except Exception as e:
        print(f"Error listing payments for job {job_id}: {e}")
        return []

# ============== STATS ==============

def get_network_stats() -> dict:
    """Calculate network stats from database"""
    try:
        workers = list_workers()
        jobs_result = supabase.table("jobs").select("*").execute()
        tasks_result = supabase.table("tasks").select("*").execute()
        payments_result = supabase.table("payments").select("*").execute()
        
        active_workers = len([w for w in workers if w.get("current_task_id") is None])
        completed_jobs = len([j for j in jobs_result.data if j.get("status") == "completed"])
        verified_tasks = len([t for t in tasks_result.data if t.get("status") == "verified"])
        total_payments = sum(p.get("amount", 0) for p in payments_result.data if p.get("status") == "released")
        
        return {
            "total_devices": len(workers),
            "active_devices": active_workers,
            "total_jobs_completed": completed_jobs,
            "total_tasks_completed": len(tasks_result.data),
            "total_verified_tasks": verified_tasks,
            "total_payments": total_payments
        }
    except Exception as e:
        print(f"Error getting network stats: {e}")
        return {
            "total_devices": 0,
            "active_devices": 0,
            "total_jobs_completed": 0,
            "total_tasks_completed": 0,
            "total_verified_tasks": 0,
            "total_payments": 0
        }
