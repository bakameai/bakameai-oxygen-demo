from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import uuid
import time
import json
import hashlib
from datetime import datetime
from enum import Enum
import asyncio

app = FastAPI(title="Oxygen Distributed Compute Network")

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== CONFIGURATION ==============
REDUNDANCY_FACTOR = 3  # Number of workers per fragment for majority voting
CONSENSUS_THRESHOLD = 2  # Minimum matching results for consensus (2 of 3)
LEASE_TIMEOUT_SECONDS = 60  # Task lease timeout
MAX_RETRIES = 3  # Maximum retries before marking task as failed
PAYOUT_PER_TASK = 0.0001  # $0.0001 per verified task

# ============== ENUMS ==============
class JobStatus(str, Enum):
    PENDING = "pending"
    DECOMPOSING = "decomposing"
    QUEUED = "queued"
    PROCESSING = "processing"
    VALIDATING = "validating"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"

class FragmentStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    COMPUTING = "computing"
    VALIDATING = "validating"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    FAILED = "failed"

class TaskStatus(str, Enum):
    PENDING = "pending"
    LEASED = "leased"
    SUBMITTED = "submitted"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"

class KernelType(str, Enum):
    IMAGE_BLUR = "image_blur"
    IMAGE_EDGE = "image_edge"
    IMAGE_GRAYSCALE = "image_grayscale"
    FILE_HASH = "file_hash"
    TEXT_WORDCOUNT = "text_wordcount"
    MATRIX_MULTIPLY = "matrix_multiply"
    IMAGE_CLASSIFY = "image_classify"
    MODEL_FINETUNE = "model_finetune"
    TEXT_EMBED = "text_embed"
    IMAGE_EMBED = "image_embed"
    VIDEO_ANALYZE = "video_analyze"

class PaymentStatus(str, Enum):
    ESCROWED = "escrowed"
    RELEASED = "released"
    REFUNDED = "refunded"

# ============== MODELS ==============

class Job(BaseModel):
    id: str
    kernel_type: KernelType
    customer_id: str
    status: JobStatus
    created_at: float
    completed_at: Optional[float] = None
    total_fragments: int = 0
    completed_fragments: int = 0
    verified_fragments: int = 0
    data_id: Optional[str] = None
    params: dict = {}
    result: Optional[dict] = None
    escrow_amount: float = 0.0
    lifecycle: list = []

class Fragment(BaseModel):
    id: str
    job_id: str
    fragment_index: int
    shard_id: str
    shard_params: dict = {}
    status: FragmentStatus
    created_at: float
    completed_at: Optional[float] = None
    retry_count: int = 0
    submissions: list = []
    consensus_hash: Optional[str] = None
    verified_result: Optional[dict] = None

class Task(BaseModel):
    id: str
    fragment_id: str
    job_id: str
    worker_id: Optional[str] = None
    status: TaskStatus
    created_at: float
    leased_at: Optional[float] = None
    submitted_at: Optional[float] = None
    lease_expires_at: Optional[float] = None
    compute_time_ms: float = 0.0
    result_hash: Optional[str] = None
    result_data: Optional[dict] = None
    payout: float = 0.0

class Worker(BaseModel):
    id: str
    name: str
    device_info: dict
    capabilities: dict = {}
    connected_at: float
    last_heartbeat: float
    total_tasks_completed: int = 0
    total_compute_time_ms: float = 0.0
    total_earnings: float = 0.0
    pending_earnings: float = 0.0
    is_active: bool = True
    verified_tasks: int = 0
    current_task_id: Optional[str] = None

class WorkerRegister(BaseModel):
    name: str
    device_info: dict
    capabilities: dict = {}

class Payment(BaseModel):
    id: str
    job_id: str
    customer_id: str
    worker_id: Optional[str] = None
    amount: float
    status: PaymentStatus
    created_at: float
    released_at: Optional[float] = None
    task_id: Optional[str] = None
    fragment_id: Optional[str] = None

# ============== IN-MEMORY DATABASE ==============
jobs_db: dict[str, Job] = {}
fragments_db: dict[str, Fragment] = {}
tasks_db: dict[str, Task] = {}
task_queue: list[str] = []
workers_db: dict[str, Worker] = {}
payments_db: dict[str, Payment] = {}
escrow_db: dict[str, float] = {}
data_db: dict[str, bytes] = {}
results_db: dict[str, bytes] = {}

network_stats = {
    "total_devices": 0,
    "total_compute_time_ms": 0.0,
    "total_jobs_completed": 0,
    "total_fragments_completed": 0,
    "total_tasks_completed": 0,
    "total_verified_tasks": 0,
    "total_payments": 0.0,
    "total_escrowed": 0.0,
    "consensus_reached": 0,
    "disagreements": 0,
    "recomputations": 0
}

# ============== CONNECTION MANAGER ==============
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.worker_connections: dict[str, WebSocket] = {}
    
    async def connect_client(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    async def connect_worker(self, websocket: WebSocket, worker_id: str):
        await websocket.accept()
        self.worker_connections[worker_id] = websocket
    
    def disconnect_client(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    def disconnect_worker(self, worker_id: str):
        if worker_id in self.worker_connections:
            del self.worker_connections[worker_id]
        if worker_id in workers_db:
            workers_db[worker_id].is_active = False
            workers_db[worker_id].current_task_id = None
            network_stats["total_devices"] = sum(1 for w in workers_db.values() if w.is_active)
    
    async def broadcast_to_clients(self, message: dict):
        for connection in self.active_connections.values():
            try:
                await connection.send_json(message)
            except:
                pass
    
    async def broadcast_to_workers(self, message: dict):
        for connection in self.worker_connections.values():
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# ============== HELPER FUNCTIONS ==============

def add_job_lifecycle_event(job: Job, event: str, details: dict = {}):
    job.lifecycle.append({"timestamp": time.time(), "event": event, "details": details})

def get_eligible_workers(kernel_type: KernelType, exclude_workers: list[str] = []) -> list[Worker]:
    eligible = []
    for worker in workers_db.values():
        if not worker.is_active or worker.id in exclude_workers or worker.current_task_id is not None:
            continue
        caps = worker.capabilities
        supported_kernels = caps.get("supported_kernels", [])
        if not supported_kernels or kernel_type.value in supported_kernels:
            eligible.append(worker)
    eligible.sort(key=lambda w: w.total_tasks_completed)
    return eligible

async def check_lease_expirations():
    current_time = time.time()
    for task in tasks_db.values():
        if task.status == TaskStatus.LEASED and task.lease_expires_at and current_time > task.lease_expires_at:
            task.status = TaskStatus.EXPIRED
            if task.worker_id and task.worker_id in workers_db:
                workers_db[task.worker_id].current_task_id = None
            fragment = fragments_db.get(task.fragment_id)
            if fragment and fragment.status == FragmentStatus.COMPUTING:
                fragment.retry_count += 1
                if fragment.retry_count < MAX_RETRIES:
                    await queue_fragment_tasks(fragment)
                    network_stats["recomputations"] += 1
                else:
                    fragment.status = FragmentStatus.FAILED

async def queue_fragment_tasks(fragment: Fragment):
    job = jobs_db.get(fragment.job_id)
    if not job:
        return
    tasks_needed = REDUNDANCY_FACTOR - len([t for t in fragment.submissions if tasks_db.get(t) and tasks_db[t].status in [TaskStatus.LEASED, TaskStatus.SUBMITTED, TaskStatus.VERIFIED]])
    for _ in range(tasks_needed):
        task = Task(id=str(uuid.uuid4()), fragment_id=fragment.id, job_id=fragment.job_id, status=TaskStatus.PENDING, created_at=time.time(), payout=PAYOUT_PER_TASK)
        tasks_db[task.id] = task
        task_queue.append(task.id)
        fragment.submissions.append(task.id)
    fragment.status = FragmentStatus.QUEUED

async def validate_fragment(fragment: Fragment):
    submitted_tasks = [tasks_db[tid] for tid in fragment.submissions if tasks_db.get(tid) and tasks_db[tid].status == TaskStatus.SUBMITTED]
    if len(submitted_tasks) < CONSENSUS_THRESHOLD:
        return
    hash_counts: dict[str, list[Task]] = {}
    for task in submitted_tasks:
        h = task.result_hash or "none"
        if h not in hash_counts:
            hash_counts[h] = []
        hash_counts[h].append(task)
    for result_hash, tasks in hash_counts.items():
        if len(tasks) >= CONSENSUS_THRESHOLD:
            fragment.status = FragmentStatus.VERIFIED
            fragment.consensus_hash = result_hash
            fragment.completed_at = time.time()
            fragment.verified_result = tasks[0].result_data
            for task in tasks:
                task.status = TaskStatus.VERIFIED
                await release_task_payment(task)
            for other_hash, other_tasks in hash_counts.items():
                if other_hash != result_hash:
                    for task in other_tasks:
                        task.status = TaskStatus.REJECTED
            network_stats["consensus_reached"] += 1
            network_stats["total_fragments_completed"] += 1
            await check_job_completion(fragment.job_id)
            return
    if len(submitted_tasks) >= REDUNDANCY_FACTOR:
        fragment.status = FragmentStatus.DISPUTED
        fragment.retry_count += 1
        network_stats["disagreements"] += 1
        if fragment.retry_count < MAX_RETRIES:
            await queue_fragment_tasks(fragment)
            network_stats["recomputations"] += 1
        else:
            fragment.status = FragmentStatus.FAILED

async def release_task_payment(task: Task):
    if not task.worker_id:
        return
    worker = workers_db.get(task.worker_id)
    job = jobs_db.get(task.job_id)
    if not worker or not job:
        return
    payment = Payment(id=str(uuid.uuid4()), job_id=task.job_id, customer_id=job.customer_id, worker_id=task.worker_id, amount=task.payout, status=PaymentStatus.RELEASED, created_at=time.time(), released_at=time.time(), task_id=task.id, fragment_id=task.fragment_id)
    payments_db[payment.id] = payment
    worker.total_earnings += task.payout
    worker.pending_earnings -= task.payout
    worker.verified_tasks += 1
    if job.id in escrow_db:
        escrow_db[job.id] -= task.payout
    network_stats["total_payments"] += task.payout
    network_stats["total_verified_tasks"] += 1

async def check_job_completion(job_id: str):
    job = jobs_db.get(job_id)
    if not job:
        return
    job_fragments = [f for f in fragments_db.values() if f.job_id == job_id]
    verified_fragments = [f for f in job_fragments if f.status == FragmentStatus.VERIFIED]
    failed_fragments = [f for f in job_fragments if f.status == FragmentStatus.FAILED]
    job.verified_fragments = len(verified_fragments)
    job.completed_fragments = len(verified_fragments) + len(failed_fragments)
    if len(verified_fragments) + len(failed_fragments) == len(job_fragments):
        if len(failed_fragments) > 0:
            job.status = JobStatus.FAILED
            add_job_lifecycle_event(job, "job_failed", {"failed_fragments": len(failed_fragments)})
        else:
            job.status = JobStatus.AGGREGATING
            add_job_lifecycle_event(job, "aggregating", {})
            job.result = aggregate_job_results(job, job_fragments)
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            network_stats["total_jobs_completed"] += 1
            add_job_lifecycle_event(job, "completed", {"result_keys": list(job.result.keys()) if job.result else []})
        await manager.broadcast_to_clients({"type": "job_completed", "job": job.model_dump()})

def aggregate_job_results(job: Job, fragments: list[Fragment]) -> dict:
    sorted_fragments = sorted(fragments, key=lambda f: f.fragment_index)
    if job.kernel_type == KernelType.FILE_HASH:
        hashes = [f.verified_result["chunk_hash"] for f in sorted_fragments if f.verified_result and "chunk_hash" in f.verified_result]
        merkle_root = hashlib.sha256("".join(hashes).encode()).hexdigest() if hashes else None
        return {"merkle_root": merkle_root, "chunk_hashes": hashes, "total_chunks": len(hashes)}
    elif job.kernel_type == KernelType.TEXT_WORDCOUNT:
        total_counts: dict[str, int] = {}
        total_words = 0
        total_lines = 0
        for f in sorted_fragments:
            if f.verified_result:
                for word, count in f.verified_result.get("word_counts", {}).items():
                    total_counts[word] = total_counts.get(word, 0) + count
                total_words += f.verified_result.get("total_words", 0)
                total_lines += f.verified_result.get("lines_processed", 0)
        sorted_words = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        return {"word_counts": dict(sorted_words), "total_unique_words": len(total_counts), "total_words": total_words, "total_lines": total_lines}
    elif job.kernel_type == KernelType.MATRIX_MULTIPLY:
        blocks = {f.shard_id: f.verified_result["block_checksum"] for f in sorted_fragments if f.verified_result and "block_checksum" in f.verified_result}
        return {"blocks": blocks, "matrix_size": job.params.get("matrix_size", 64), "status": "completed"}
    elif job.kernel_type in [KernelType.IMAGE_BLUR, KernelType.IMAGE_EDGE, KernelType.IMAGE_GRAYSCALE]:
        tiles = {f.shard_id: {"verified": f.status == FragmentStatus.VERIFIED, "hash": f.consensus_hash} for f in sorted_fragments}
        return {"tiles": tiles, "grid_size": job.params.get("grid_size", 4), "status": "completed"}
    elif job.kernel_type == KernelType.IMAGE_CLASSIFY:
        classifications = []
        for f in sorted_fragments:
            if f.verified_result:
                classifications.append({
                    "image_index": f.shard_params.get("image_index", 0),
                    "predictions": f.verified_result.get("predictions", []),
                    "top_class": f.verified_result.get("top_class", "unknown"),
                    "confidence": f.verified_result.get("confidence", 0)
                })
        return {"classifications": classifications, "status": "completed"}
    elif job.kernel_type == KernelType.MODEL_FINETUNE:
        # Aggregate training results from all rounds
        training_rounds = []
        total_loss_improvement = 0
        final_accuracy = 0
        for f in sorted_fragments:
            if f.verified_result:
                round_result = {
                    "round": f.shard_params.get("round_index", 0),
                    "loss_before": f.verified_result.get("loss_before", 0),
                    "loss_after": f.verified_result.get("loss_after", 0),
                    "accuracy": f.verified_result.get("accuracy", 0),
                    "samples_trained": f.verified_result.get("samples_trained", 0)
                }
                training_rounds.append(round_result)
                total_loss_improvement += (round_result["loss_before"] - round_result["loss_after"])
                final_accuracy = max(final_accuracy, round_result["accuracy"])
        return {
            "training_rounds": training_rounds,
            "total_rounds": len(training_rounds),
            "total_loss_improvement": total_loss_improvement,
            "final_accuracy": final_accuracy,
            "status": "completed"
        }
    elif job.kernel_type == KernelType.TEXT_EMBED:
        # Aggregate text embeddings from all batches
        all_embeddings = []
        all_items = []
        for f in sorted_fragments:
            if f.verified_result:
                embeddings = f.verified_result.get("embeddings", [])
                items = f.verified_result.get("items", [])
                all_embeddings.extend(embeddings)
                all_items.extend(items)
        return {
            "embeddings": all_embeddings,
            "items": all_items,
            "total_items": len(all_items),
            "embedding_dim": len(all_embeddings[0]) if all_embeddings else 0,
            "status": "completed"
        }
    elif job.kernel_type == KernelType.IMAGE_EMBED:
        # Aggregate image embeddings from all batches
        all_embeddings = []
        all_image_ids = []
        for f in sorted_fragments:
            if f.verified_result:
                embeddings = f.verified_result.get("embeddings", [])
                image_ids = f.verified_result.get("image_ids", [])
                all_embeddings.extend(embeddings)
                all_image_ids.extend(image_ids)
        return {
            "embeddings": all_embeddings,
            "image_ids": all_image_ids,
            "total_images": len(all_image_ids),
            "embedding_dim": len(all_embeddings[0]) if all_embeddings else 0,
            "status": "completed"
        }
    elif job.kernel_type == KernelType.VIDEO_ANALYZE:
        # Aggregate video analysis results from all frame batches
        all_detections = []
        frames_with_detections = 0
        detection_counts = {}
        for f in sorted_fragments:
            if f.verified_result:
                detections = f.verified_result.get("detections", [])
                all_detections.extend(detections)
                for det in detections:
                    label = det.get("label", "unknown")
                    detection_counts[label] = detection_counts.get(label, 0) + 1
                    if det.get("confidence", 0) > 0.5:
                        frames_with_detections += 1
        # Get highlight frames (frames with high-confidence detections)
        highlights = [d for d in all_detections if d.get("confidence", 0) > 0.7]
        highlights = sorted(highlights, key=lambda x: x.get("confidence", 0), reverse=True)[:20]
        return {
            "detections": all_detections,
            "highlights": highlights,
            "detection_counts": detection_counts,
            "total_frames_analyzed": job.params.get("num_frames", 0),
            "frames_with_detections": frames_with_detections,
            "status": "completed"
        }
    return {"status": "completed", "fragments": len(sorted_fragments)}

# ============== API ENDPOINTS ==============

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    data_id = str(uuid.uuid4())
    data_db[data_id] = contents
    filename = file.filename or ""
    file_type = "binary"
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
        file_type = "image"
    elif filename.lower().endswith(('.txt', '.csv', '.json', '.md')):
        file_type = "text"
    return {"data_id": data_id, "size_bytes": len(contents), "filename": filename, "file_type": file_type}

@app.get("/data/{data_id}")
async def get_data(data_id: str):
    if data_id not in data_db:
        raise HTTPException(status_code=404, detail="Data not found")
    return Response(content=data_db[data_id], media_type="application/octet-stream")

@app.get("/data/{data_id}/range")
async def get_data_range(data_id: str, start: int = 0, end: Optional[int] = None):
    if data_id not in data_db:
        raise HTTPException(status_code=404, detail="Data not found")
    data = data_db[data_id]
    if end is None:
        end = len(data)
    return Response(content=data[start:end], media_type="application/octet-stream")

class JobCreateRequest(BaseModel):
    inline_data: Optional[str] = None

@app.post("/jobs/create")
async def create_job(data_id: str, kernel_type: KernelType, customer_id: str = "demo-customer", params: str = "{}", inline_data: Optional[str] = None, body: Optional[JobCreateRequest] = None):
    # If inline_data is provided (from query param or body), store it directly (handles multi-instance deployments)
    # Prefer body over query param to avoid URL length limits
    actual_inline_data = (body.inline_data if body and body.inline_data else None) or inline_data
    if actual_inline_data is not None:
        data_db[data_id] = actual_inline_data.encode('utf-8')
    # Matrix multiplication doesn't require uploaded data - it generates random matrices
    if kernel_type != KernelType.MATRIX_MULTIPLY and data_id not in data_db:
        raise HTTPException(status_code=404, detail="Data not found")
    try:
        job_params = json.loads(params)
    except:
        job_params = {}
    job_id = str(uuid.uuid4())
    # For matrix multiply, create a placeholder data entry
    if kernel_type == KernelType.MATRIX_MULTIPLY:
        data_db[data_id] = b"matrix_placeholder"
    data = data_db[data_id]
    data_size = len(data)
    job = Job(id=job_id, kernel_type=kernel_type, customer_id=customer_id, status=JobStatus.PENDING, created_at=time.time(), data_id=data_id, params=job_params, lifecycle=[])
    jobs_db[job_id] = job
    add_job_lifecycle_event(job, "job_created", {"kernel_type": kernel_type.value, "data_size": data_size})
    job.status = JobStatus.DECOMPOSING
    add_job_lifecycle_event(job, "decomposing", {})
    fragments_to_create = []
    if kernel_type in [KernelType.IMAGE_BLUR, KernelType.IMAGE_EDGE, KernelType.IMAGE_GRAYSCALE]:
        grid_size = job_params.get("grid_size", 4)
        if grid_size < 2 or grid_size > 8:
            grid_size = 4
        for row in range(grid_size):
            for col in range(grid_size):
                fragments_to_create.append({"shard_id": f"tile_{row}_{col}", "shard_params": {"row": row, "col": col, "grid_size": grid_size}})
        job_params["grid_size"] = grid_size
    elif kernel_type == KernelType.FILE_HASH:
        chunk_size = job_params.get("chunk_size", 64 * 1024)
        num_chunks = max(1, (data_size + chunk_size - 1) // chunk_size)
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, data_size)
            fragments_to_create.append({"shard_id": f"chunk_{i}", "shard_params": {"chunk_index": i, "start_byte": start, "end_byte": end, "total_chunks": num_chunks}})
        job_params["chunk_size"] = chunk_size
        job_params["total_size"] = data_size
    elif kernel_type == KernelType.TEXT_WORDCOUNT:
        text = data.decode('utf-8', errors='ignore')
        lines = text.split('\n')
        lines_per_chunk = job_params.get("lines_per_chunk", 100)
        num_chunks = max(1, (len(lines) + lines_per_chunk - 1) // lines_per_chunk)
        for i in range(num_chunks):
            start_line = i * lines_per_chunk
            end_line = min(start_line + lines_per_chunk, len(lines))
            fragments_to_create.append({"shard_id": f"lines_{start_line}_{end_line}", "shard_params": {"chunk_index": i, "start_line": start_line, "end_line": end_line, "total_lines": len(lines)}})
        job_params["lines_per_chunk"] = lines_per_chunk
        job_params["total_lines"] = len(lines)
    elif kernel_type == KernelType.MATRIX_MULTIPLY:
        matrix_size = job_params.get("matrix_size", 64)
        block_size = job_params.get("block_size", 16)
        num_blocks = (matrix_size + block_size - 1) // block_size
        for row in range(num_blocks):
            for col in range(num_blocks):
                for k in range(num_blocks):
                    fragments_to_create.append({"shard_id": f"block_{row}_{col}_{k}", "shard_params": {"block_row": row, "block_col": col, "block_k": k, "block_size": block_size, "matrix_size": matrix_size}})
        job_params["matrix_size"] = matrix_size
        job_params["block_size"] = block_size
    elif kernel_type == KernelType.IMAGE_CLASSIFY:
        # For image classification, each image is one fragment
        # The data_id points to the uploaded image
        # We create a single fragment for the image classification task
        fragments_to_create.append({"shard_id": "image_0", "shard_params": {"image_index": 0}})
    elif kernel_type == KernelType.MODEL_FINETUNE:
        # For model fine-tuning, we create training rounds
        # Each round trains on a batch of data and returns weight updates
        num_rounds = job_params.get("num_rounds", 5)  # Number of training rounds
        epochs_per_round = job_params.get("epochs_per_round", 1)  # Epochs per worker per round
        for round_idx in range(num_rounds):
            fragments_to_create.append({
                "shard_id": f"round_{round_idx}",
                "shard_params": {
                    "round_index": round_idx,
                    "epochs": epochs_per_round,
                    "total_rounds": num_rounds
                }
            })
        job_params["num_rounds"] = num_rounds
        job_params["epochs_per_round"] = epochs_per_round
    elif kernel_type == KernelType.TEXT_EMBED:
        # For text embeddings, split text into items (lines or JSON array)
        # Each item gets an embedding generated
        text = data.decode('utf-8', errors='ignore')
        items = []
        # Try to parse as JSON array first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                items = [str(item) if isinstance(item, dict) else item for item in parsed]
            else:
                items = text.strip().split('\n')
        except:
            items = text.strip().split('\n')
        items = [item.strip() for item in items if item.strip()]
        items_per_batch = job_params.get("items_per_batch", 10)
        num_batches = max(1, (len(items) + items_per_batch - 1) // items_per_batch)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * items_per_batch
            end_idx = min(start_idx + items_per_batch, len(items))
            fragments_to_create.append({
                "shard_id": f"batch_{batch_idx}",
                "shard_params": {
                    "batch_index": batch_idx,
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "total_items": len(items)
                }
            })
        job_params["total_items"] = len(items)
        job_params["items_per_batch"] = items_per_batch
    elif kernel_type == KernelType.IMAGE_EMBED:
        # For image embeddings, the data contains multiple images (JSON with base64 images)
        # Each image gets an embedding generated
        text = data.decode('utf-8', errors='ignore')
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                num_images = len(parsed)
            elif isinstance(parsed, dict) and "images" in parsed:
                num_images = len(parsed["images"])
            else:
                num_images = 1
        except:
            num_images = 1
        images_per_batch = job_params.get("images_per_batch", 5)
        num_batches = max(1, (num_images + images_per_batch - 1) // images_per_batch)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * images_per_batch
            end_idx = min(start_idx + images_per_batch, num_images)
            fragments_to_create.append({
                "shard_id": f"batch_{batch_idx}",
                "shard_params": {
                    "batch_index": batch_idx,
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "total_images": num_images
                }
            })
        job_params["total_images"] = num_images
        job_params["images_per_batch"] = images_per_batch
    elif kernel_type == KernelType.VIDEO_ANALYZE:
        # For video analysis, we process frames
        # The data contains video metadata or frame count
        # Each batch of frames is a fragment
        num_frames = job_params.get("num_frames", 100)  # Total frames to analyze
        frames_per_batch = job_params.get("frames_per_batch", 10)
        num_batches = max(1, (num_frames + frames_per_batch - 1) // frames_per_batch)
        for batch_idx in range(num_batches):
            start_frame = batch_idx * frames_per_batch
            end_frame = min(start_frame + frames_per_batch, num_frames)
            fragments_to_create.append({
                "shard_id": f"frames_{start_frame}_{end_frame}",
                "shard_params": {
                    "batch_index": batch_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "total_frames": num_frames
                }
            })
        job_params["num_frames"] = num_frames
        job_params["frames_per_batch"] = frames_per_batch
    job.params = job_params
    job.total_fragments = len(fragments_to_create)
    add_job_lifecycle_event(job, "decomposed", {"fragment_count": len(fragments_to_create)})
    escrow_amount = len(fragments_to_create) * PAYOUT_PER_TASK * REDUNDANCY_FACTOR
    job.escrow_amount = escrow_amount
    escrow_db[job_id] = escrow_amount
    network_stats["total_escrowed"] += escrow_amount
    add_job_lifecycle_event(job, "escrow_created", {"amount": escrow_amount})
    for i, frag_info in enumerate(fragments_to_create):
        fragment = Fragment(id=str(uuid.uuid4()), job_id=job_id, fragment_index=i, shard_id=frag_info["shard_id"], shard_params=frag_info["shard_params"], status=FragmentStatus.PENDING, created_at=time.time())
        fragments_db[fragment.id] = fragment
        await queue_fragment_tasks(fragment)
    job.status = JobStatus.QUEUED
    add_job_lifecycle_event(job, "queued", {"tasks_created": len(fragments_to_create) * REDUNDANCY_FACTOR})
    await manager.broadcast_to_workers({"type": "new_job", "job_id": job_id, "kernel_type": kernel_type.value, "fragment_count": len(fragments_to_create), "tasks_per_fragment": REDUNDANCY_FACTOR})
    await manager.broadcast_to_clients({"type": "job_created", "job": job.model_dump()})
    return job

@app.get("/tasks/next")
async def get_next_task(worker_id: str):
    if worker_id not in workers_db:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker = workers_db[worker_id]
    if worker.current_task_id:
        return {"task": None, "message": "Worker already has an active task"}
    await check_lease_expirations()
    for task_id in list(task_queue):
        task = tasks_db.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            if task_id in task_queue:
                task_queue.remove(task_id)
            continue
        fragment = fragments_db.get(task.fragment_id)
        job = jobs_db.get(task.job_id)
        if not fragment or not job:
            continue
        eligible = get_eligible_workers(job.kernel_type, exclude_workers=[])
        if worker not in eligible:
            continue
        task.status = TaskStatus.LEASED
        task.worker_id = worker_id
        task.leased_at = time.time()
        task.lease_expires_at = time.time() + LEASE_TIMEOUT_SECONDS
        worker.current_task_id = task_id
        worker.pending_earnings += task.payout
        if task_id in task_queue:
            task_queue.remove(task_id)
        fragment.status = FragmentStatus.COMPUTING
        return {"task": task.model_dump(), "fragment": fragment.model_dump(), "kernel_type": job.kernel_type.value, "data_id": job.data_id, "job_params": job.params, "lease_seconds": LEASE_TIMEOUT_SECONDS}
    return {"task": None, "message": "No tasks available"}

@app.post("/tasks/{task_id}/submit")
async def submit_task_result(task_id: str, worker_id: str = Form(...), compute_time_ms: float = Form(...), result_hash: str = Form(...), result_data: str = Form("{}"), file: Optional[UploadFile] = File(None)):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    if worker_id not in workers_db:
        raise HTTPException(status_code=404, detail="Worker not found")
    task = tasks_db[task_id]
    worker = workers_db[worker_id]
    if task.worker_id != worker_id:
        raise HTTPException(status_code=403, detail="Task not assigned to this worker")
    if task.status != TaskStatus.LEASED:
        raise HTTPException(status_code=400, detail=f"Task not in leased state: {task.status}")
    if file:
        contents = await file.read()
        fragment = fragments_db.get(task.fragment_id)
        if fragment:
            result_key = f"{task.job_id}:{fragment.shard_id}:{task_id}"
            results_db[result_key] = contents
    try:
        parsed_result = json.loads(result_data)
    except:
        parsed_result = {}
    task.status = TaskStatus.SUBMITTED
    task.submitted_at = time.time()
    task.compute_time_ms = compute_time_ms
    task.result_hash = result_hash
    task.result_data = parsed_result
    worker.current_task_id = None
    worker.total_tasks_completed += 1
    worker.total_compute_time_ms += compute_time_ms
    worker.last_heartbeat = time.time()
    network_stats["total_tasks_completed"] += 1
    network_stats["total_compute_time_ms"] += compute_time_ms
    fragment = fragments_db.get(task.fragment_id)
    if fragment:
        fragment.status = FragmentStatus.VALIDATING
        await validate_fragment(fragment)
    job = jobs_db.get(task.job_id)
    if job:
        job.status = JobStatus.PROCESSING
        await manager.broadcast_to_clients({"type": "task_submitted", "task": task.model_dump(), "fragment_id": task.fragment_id, "job_progress": {"job_id": task.job_id, "verified_fragments": job.verified_fragments, "total_fragments": job.total_fragments}})
    return {"status": "submitted", "awaiting_validation": True, "message": "Result submitted for validation"}

# Legacy endpoints
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    return await upload_data(file)

@app.get("/images/{image_id}")
async def get_image(image_id: str):
    if image_id not in data_db:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=data_db[image_id], media_type="image/png")

@app.post("/jobs/create-image-job")
async def create_image_job(image_id: str, job_type: str = "image_blur", customer_id: str = "demo-customer", grid_size: int = 4):
    kernel_map = {"image_blur": KernelType.IMAGE_BLUR, "image_edge": KernelType.IMAGE_EDGE, "image_grayscale": KernelType.IMAGE_GRAYSCALE}
    kernel_type = kernel_map.get(job_type, KernelType.IMAGE_BLUR)
    return await create_job(data_id=image_id, kernel_type=kernel_type, customer_id=customer_id, params=json.dumps({"grid_size": grid_size}))

@app.post("/jobs/{job_id}/submit-tile")
async def submit_tile_result(job_id: str, worker_id: str = Form(...), tile_row: int = Form(...), tile_col: int = Form(...), compute_time_ms: float = Form(...), result_hash: str = Form(...), file: UploadFile = File(...)):
    shard_id = f"tile_{tile_row}_{tile_col}"
    task = None
    for t in tasks_db.values():
        if t.job_id == job_id and t.worker_id == worker_id:
            fragment = fragments_db.get(t.fragment_id)
            if fragment and fragment.shard_id == shard_id:
                task = t
                break
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    contents = await file.read()
    result_key = f"{job_id}:{shard_id}:{task.id}"
    results_db[result_key] = contents
    return await submit_task_result(task_id=task.id, worker_id=worker_id, compute_time_ms=compute_time_ms, result_hash=result_hash, result_data="{}", file=None)

@app.post("/jobs/{job_id}/submit-result")
async def submit_job_result(job_id: str, worker_id: str = Form(...), task_id: str = Form(...), compute_time_ms: float = Form(...), result_hash: str = Form(...), result_data: str = Form("{}"), file: Optional[UploadFile] = File(None)):
    return await submit_task_result(task_id=task_id, worker_id=worker_id, compute_time_ms=compute_time_ms, result_hash=result_hash, result_data=result_data, file=file)

@app.get("/jobs/{job_id}/result-tile/{row}/{col}")
async def get_result_tile(job_id: str, row: int, col: int):
    shard_id = f"tile_{row}_{col}"
    for key in results_db:
        if key.startswith(f"{job_id}:{shard_id}:"):
            return Response(content=results_db[key], media_type="image/png")
    raise HTTPException(status_code=404, detail="Processed tile not found")

@app.get("/jobs/{job_id}/source-image")
async def get_job_source_image(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs_db[job_id]
    if not job.data_id or job.data_id not in data_db:
        raise HTTPException(status_code=404, detail="Data not found")
    return Response(content=data_db[job.data_id], media_type="image/png")

@app.get("/jobs")
async def list_jobs(customer_id: Optional[str] = None):
    if customer_id:
        return [j.model_dump() for j in jobs_db.values() if j.customer_id == customer_id]
    return [j.model_dump() for j in jobs_db.values()]

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs_db[job_id]
    job_fragments = [f for f in fragments_db.values() if f.job_id == job_id]
    job_tasks = [t for t in tasks_db.values() if t.job_id == job_id]
    return {**job.model_dump(), "fragments": [f.model_dump() for f in sorted(job_fragments, key=lambda x: x.fragment_index)], "tasks": [t.model_dump() for t in job_tasks], "escrow_remaining": escrow_db.get(job_id, 0)}

# ============== WORKER ENDPOINTS ==============

@app.post("/workers/register")
async def register_worker(worker_data: WorkerRegister):
    worker_id = str(uuid.uuid4())
    worker = Worker(id=worker_id, name=worker_data.name, device_info=worker_data.device_info, capabilities=worker_data.capabilities, connected_at=time.time(), last_heartbeat=time.time(), is_active=True)
    workers_db[worker_id] = worker
    network_stats["total_devices"] = sum(1 for w in workers_db.values() if w.is_active)
    await manager.broadcast_to_clients({"type": "worker_joined", "worker_id": worker_id, "name": worker_data.name, "capabilities": worker_data.capabilities})
    return worker

@app.post("/workers/{worker_id}/heartbeat")
async def worker_heartbeat(worker_id: str):
    if worker_id not in workers_db:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker = workers_db[worker_id]
    worker.last_heartbeat = time.time()
    worker.is_active = True
    network_stats["total_devices"] = sum(1 for w in workers_db.values() if w.is_active)
    return {"status": "ok", "current_task": worker.current_task_id}

@app.post("/workers/{worker_id}/disconnect")
async def disconnect_worker(worker_id: str):
    if worker_id not in workers_db:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker = workers_db[worker_id]
    worker.is_active = False
    if worker.current_task_id:
        task = tasks_db.get(worker.current_task_id)
        if task and task.status == TaskStatus.LEASED:
            task.status = TaskStatus.EXPIRED
            task_queue.append(task.id)
        worker.current_task_id = None
    network_stats["total_devices"] = sum(1 for w in workers_db.values() if w.is_active)
    await manager.broadcast_to_clients({"type": "worker_left", "worker_id": worker_id})
    return {"status": "disconnected"}

@app.get("/workers")
async def list_workers():
    return [w.model_dump() for w in workers_db.values()]

@app.get("/workers/{worker_id}")
async def get_worker(worker_id: str):
    if worker_id not in workers_db:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker = workers_db[worker_id]
    worker_tasks = [t for t in tasks_db.values() if t.worker_id == worker_id]
    worker_payments = [p for p in payments_db.values() if p.worker_id == worker_id]
    return {**worker.model_dump(), "tasks": [t.model_dump() for t in worker_tasks], "payments": [p.model_dump() for p in worker_payments]}

# ============== STATS ENDPOINTS ==============

@app.get("/stats")
async def get_stats():
    active_workers = sum(1 for w in workers_db.values() if w.is_active)
    total_gflops = network_stats["total_compute_time_ms"] * 0.001 * 200
    return {**network_stats, "active_workers": active_workers, "total_workers": len(workers_db), "total_jobs": len(jobs_db), "pending_jobs": sum(1 for j in jobs_db.values() if j.status in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.PROCESSING]), "completed_jobs": sum(1 for j in jobs_db.values() if j.status == JobStatus.COMPLETED), "total_fragments": len(fragments_db), "pending_fragments": sum(1 for f in fragments_db.values() if f.status in [FragmentStatus.PENDING, FragmentStatus.QUEUED, FragmentStatus.COMPUTING]), "verified_fragments": sum(1 for f in fragments_db.values() if f.status == FragmentStatus.VERIFIED), "total_tasks": len(tasks_db), "pending_tasks": len(task_queue), "leased_tasks": sum(1 for t in tasks_db.values() if t.status == TaskStatus.LEASED), "total_gflops": total_gflops, "milestones": [{"name": "1 Exaflop", "target_gflops": 1e9, "achieved": total_gflops >= 1e9}, {"name": "10 Exaflops", "target_gflops": 1e10, "achieved": total_gflops >= 1e10}, {"name": "100 Exaflops", "target_gflops": 1e11, "achieved": total_gflops >= 1e11}, {"name": "1000 Exaflops", "target_gflops": 1e12, "achieved": total_gflops >= 1e12}]}

# ============== SYSTEM MONITORING ENDPOINTS ==============

@app.get("/system/queue")
async def get_queue_status():
    pending_tasks = [tasks_db[tid].model_dump() for tid in task_queue if tid in tasks_db]
    leased_tasks = [t.model_dump() for t in tasks_db.values() if t.status == TaskStatus.LEASED]
    return {"pending_count": len(pending_tasks), "leased_count": len(leased_tasks), "pending_tasks": pending_tasks[:50], "leased_tasks": leased_tasks[:50], "lease_timeout_seconds": LEASE_TIMEOUT_SECONDS, "max_retries": MAX_RETRIES}

@app.get("/system/validation")
async def get_validation_status():
    validating_fragments = [f.model_dump() for f in fragments_db.values() if f.status == FragmentStatus.VALIDATING]
    disputed_fragments = [f.model_dump() for f in fragments_db.values() if f.status == FragmentStatus.DISPUTED]
    return {"validating_count": len(validating_fragments), "disputed_count": len(disputed_fragments), "validating_fragments": validating_fragments[:50], "disputed_fragments": disputed_fragments[:50], "consensus_threshold": CONSENSUS_THRESHOLD, "redundancy_factor": REDUNDANCY_FACTOR, "total_consensus_reached": network_stats["consensus_reached"], "total_disagreements": network_stats["disagreements"], "total_recomputations": network_stats["recomputations"]}

@app.get("/system/escrow")
async def get_escrow_status():
    escrow_entries = [{"job_id": jid, "amount": amt, "job_status": jobs_db[jid].status.value if jid in jobs_db else "unknown"} for jid, amt in escrow_db.items()]
    released_payments = [p.model_dump() for p in payments_db.values() if p.status == PaymentStatus.RELEASED]
    return {"total_escrowed": network_stats["total_escrowed"], "total_released": network_stats["total_payments"], "active_escrows": escrow_entries, "recent_payments": released_payments[-50:], "payout_per_task": PAYOUT_PER_TASK}

@app.get("/system/workers")
async def get_workers_status():
    workers_list = []
    for w in workers_db.values():
        workers_list.append({**w.model_dump(), "current_task": tasks_db.get(w.current_task_id).model_dump() if w.current_task_id and w.current_task_id in tasks_db else None})
    return {"total_workers": len(workers_db), "active_workers": sum(1 for w in workers_db.values() if w.is_active), "workers": workers_list}

# ============== WEBSOCKET ENDPOINTS ==============

@app.websocket("/ws/client/{client_id}")
async def websocket_client(websocket: WebSocket, client_id: str):
    await manager.connect_client(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "subscribe_job":
                    job_id = message.get("job_id")
                    if job_id and job_id in jobs_db:
                        job = jobs_db[job_id]
                        fragments = [f.model_dump() for f in fragments_db.values() if f.job_id == job_id]
                        await websocket.send_json({"type": "job_update", "job": job.model_dump(), "fragments": fragments})
            except:
                pass
    except WebSocketDisconnect:
        manager.disconnect_client(client_id)

@app.websocket("/ws/worker/{worker_id}")
async def websocket_worker(websocket: WebSocket, worker_id: str):
    # Auto-register worker if not found (handles multi-instance deployments)
    if worker_id not in workers_db:
        workers_db[worker_id] = Worker(
            id=worker_id,
            name=f"Worker-{worker_id[:8]}",
            device_info={},
            capabilities={"supported_kernels": ["text_embed", "image_embed", "video_analyze"]},
            connected_at=time.time(),
            last_heartbeat=time.time()
        )
    await manager.connect_worker(websocket, worker_id)
    workers_db[worker_id].is_active = True
    workers_db[worker_id].last_heartbeat = time.time()
    network_stats["total_devices"] = sum(1 for w in workers_db.values() if w.is_active)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "heartbeat":
                    workers_db[worker_id].last_heartbeat = time.time()
                    await websocket.send_json({"type": "heartbeat_ack", "current_task": workers_db[worker_id].current_task_id})
                elif message.get("type") == "request_task":
                    result = await get_next_task(worker_id)
                    await websocket.send_json({"type": "task_assignment", **result})
            except:
                pass
    except WebSocketDisconnect:
        manager.disconnect_worker(worker_id)
