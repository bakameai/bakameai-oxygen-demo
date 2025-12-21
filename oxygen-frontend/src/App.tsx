import { useState, useEffect, useRef, useCallback } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Cpu, DollarSign, Zap, Users, Play, Square, Activity, TrendingUp, Server, Wallet } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface Job {
  id: string
  job_type: string
  status: string
  payment_amount: number
  created_at: number
  completed_at?: number
  total_tasks: number
  completed_tasks: number
  result?: Record<string, unknown>
}

interface Worker {
  id: string
  name: string
  device_info: Record<string, unknown>
  total_tasks_completed: number
  total_flops_contributed: number
  total_earnings: number
  is_active: boolean
}

interface Task {
  id: string
  job_id: string
  task_index: number
  input_chunk: Record<string, unknown>
  status: string
  payout: number
}

interface NetworkStats {
  total_devices: number
  total_gflops: number
  total_jobs_completed: number
  total_tasks_completed: number
  total_payments: number
  milestones: Array<{
    name: string
    target_devices: number
    target_gflops: number
    capability: string
  }>
  active_workers: Worker[]
  recent_payments: Array<{
    id: string
    worker_id: string
    amount: number
    timestamp: number
  }>
}

function formatNumber(num: number): string {
  if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T'
  if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B'
  if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M'
  if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K'
  return num.toFixed(2)
}

function CustomerDashboard() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [jobType, setJobType] = useState('matrix_multiply')
  const [paymentAmount, setPaymentAmount] = useState('10')
  const [matrixSize, setMatrixSize] = useState('100')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/jobs?customer_id=demo-customer`)
      const data = await res.json()
      setJobs(data)
    } catch (err) {
      console.error('Failed to fetch jobs:', err)
    }
  }, [])

  useEffect(() => {
    fetchJobs()
    
    const clientId = `customer-${Date.now()}`
    const wsProtocol = API_URL.startsWith('https') ? 'wss' : 'ws'
    const wsHost = API_URL.replace(/^https?:\/\//, '')
    const wsUrl = `${wsProtocol}://${wsHost}/ws/client/${clientId}`
    const ws = new WebSocket(wsUrl)
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'job_created' || data.type === 'job_completed' || data.type === 'task_completed') {
        fetchJobs()
      }
    }
    
    wsRef.current = ws
    return () => ws.close()
  }, [fetchJobs])

  const submitJob = async () => {
    setIsSubmitting(true)
    try {
      const inputData = jobType === 'matrix_multiply' 
        ? { matrix_size: parseInt(matrixSize) }
        : jobType === 'image_blur'
        ? { grid_size: 4, blur_radius: 5, width: 512, height: 512 }
        : { numbers: Array.from({ length: 100 }, (_, i) => 10000 + i) }

      const res = await fetch(`${API_URL}/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_type: jobType,
          input_data: inputData,
          customer_id: 'demo-customer',
          payment_amount: parseFloat(paymentAmount)
        })
      })
      
      if (res.ok) {
        fetchJobs()
      }
    } catch (err) {
      console.error('Failed to submit job:', err)
    }
    setIsSubmitting(false)
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Play className="h-5 w-5" />
            Submit Compute Job
          </CardTitle>
          <CardDescription>
            Submit a job to be processed by the distributed network
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label>Job Type</Label>
              <Select value={jobType} onValueChange={setJobType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="matrix_multiply">Matrix Multiplication</SelectItem>
                  <SelectItem value="image_blur">Image Processing (Blur)</SelectItem>
                  <SelectItem value="prime_factorization">Prime Factorization</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            {jobType === 'matrix_multiply' && (
              <div className="space-y-2">
                <Label>Matrix Size</Label>
                <Input 
                  type="number" 
                  value={matrixSize} 
                  onChange={(e) => setMatrixSize(e.target.value)}
                  min="10"
                  max="1000"
                />
              </div>
            )}
            
            <div className="space-y-2">
              <Label>Payment Amount ($)</Label>
              <Input 
                type="number" 
                value={paymentAmount} 
                onChange={(e) => setPaymentAmount(e.target.value)}
                min="1"
                step="0.5"
              />
            </div>
          </div>
          
          <Button onClick={submitJob} disabled={isSubmitting} className="w-full">
            {isSubmitting ? 'Submitting...' : 'Submit Job'}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Your Jobs</CardTitle>
          <CardDescription>Track the progress of your submitted jobs</CardDescription>
        </CardHeader>
        <CardContent>
          {jobs.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">No jobs submitted yet</p>
          ) : (
            <div className="space-y-4">
              {jobs.map((job) => (
                <div key={job.id} className="border rounded-lg p-4 space-y-3">
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-medium">{job.job_type.replace('_', ' ').toUpperCase()}</h4>
                      <p className="text-sm text-muted-foreground">ID: {job.id.slice(0, 8)}...</p>
                    </div>
                    <Badge variant={job.status === 'completed' ? 'default' : job.status === 'processing' ? 'secondary' : 'outline'}>
                      {job.status}
                    </Badge>
                  </div>
                  
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Progress</span>
                      <span>{job.completed_tasks} / {job.total_tasks} tasks</span>
                    </div>
                    <Progress value={(job.completed_tasks / job.total_tasks) * 100} />
                  </div>
                  
                  <div className="flex justify-between text-sm text-muted-foreground">
                    <span>Payment: ${job.payment_amount.toFixed(2)}</span>
                    <span>Created: {new Date(job.created_at * 1000).toLocaleTimeString()}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function WorkerPortal() {
  const [worker, setWorker] = useState<Worker | null>(null)
  const [isWorking, setIsWorking] = useState(false)
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  const [taskLog, setTaskLog] = useState<string[]>([])
  const [workerName, setWorkerName] = useState(`Worker-${Math.random().toString(36).slice(2, 8)}`)
  const wsRef = useRef<WebSocket | null>(null)
  const workIntervalRef = useRef<number | null>(null)

  const addLog = (message: string) => {
    setTaskLog(prev => [...prev.slice(-50), `[${new Date().toLocaleTimeString()}] ${message}`])
  }

  const registerWorker = async () => {
    try {
      const deviceInfo = {
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        cores: navigator.hardwareConcurrency || 4,
        memory: (navigator as Navigator & { deviceMemory?: number }).deviceMemory || 8,
        gpu: 'WebGPU Compatible'
      }

      const res = await fetch(`${API_URL}/workers/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: workerName,
          device_info: deviceInfo
        })
      })

      if (res.ok) {
        const data = await res.json()
        setWorker(data)
        addLog(`Registered as worker: ${data.id.slice(0, 8)}...`)
        return data
      }
    } catch (err) {
      console.error('Failed to register worker:', err)
      addLog('Failed to register worker')
    }
    return null
  }

  const performCompute = (task: Task): { result: Record<string, unknown>, flops: number } => {
    const startTime = performance.now()
    let operations = 0
    
    if (task.input_chunk.matrix_size) {
      const size = task.input_chunk.matrix_size as number
      const startRow = task.input_chunk.start_row as number
      const endRow = task.input_chunk.end_row as number
      const rows = endRow - startRow
      
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < size; j++) {
          let sum = 0
          for (let k = 0; k < size; k++) {
            sum += Math.random() * Math.random()
            operations += 2
          }
        }
      }
      
      const elapsed = performance.now() - startTime
      const flops = operations / (elapsed / 1000)
      
      return {
        result: { 
          chunk_id: task.input_chunk.chunk_id,
          rows_processed: rows,
          operations: operations
        },
        flops: flops
      }
    }
    
    if (task.input_chunk.blur_radius !== undefined) {
      const width = task.input_chunk.image_width as number
      const height = task.input_chunk.image_height as number
      const gridSize = task.input_chunk.grid_size as number
      const chunkWidth = width / gridSize
      const chunkHeight = height / gridSize
      const blurRadius = task.input_chunk.blur_radius as number
      
      for (let y = 0; y < chunkHeight; y++) {
        for (let x = 0; x < chunkWidth; x++) {
          for (let dy = -blurRadius; dy <= blurRadius; dy++) {
            for (let dx = -blurRadius; dx <= blurRadius; dx++) {
              operations += 3
            }
          }
        }
      }
      
      const elapsed = performance.now() - startTime
      const flops = operations / (elapsed / 1000)
      
      return {
        result: {
          chunk_id: task.input_chunk.chunk_id,
          row: task.input_chunk.row,
          col: task.input_chunk.col,
          pixels_processed: chunkWidth * chunkHeight,
          blur_applied: true
        },
        flops: flops
      }
    }
    
    if (task.input_chunk.numbers) {
      const numbers = task.input_chunk.numbers as number[]
      const factors: Record<number, number[]> = {}
      
      for (const num of numbers) {
        factors[num] = []
        let n = num
        for (let i = 2; i <= Math.sqrt(n); i++) {
          while (n % i === 0) {
            factors[num].push(i)
            n = n / i
            operations += 10
          }
        }
        if (n > 1) factors[num].push(n)
      }
      
      const elapsed = performance.now() - startTime
      const flops = operations / (elapsed / 1000)
      
      return {
        result: {
          chunk_id: task.input_chunk.chunk_id,
          factors: factors,
          numbers_processed: numbers.length
        },
        flops: flops
      }
    }
    
    return { result: { status: 'completed' }, flops: 1000000 }
  }

  const startWorking = async () => {
    let currentWorker = worker
    if (!currentWorker) {
      currentWorker = await registerWorker()
      if (!currentWorker) return
    }

    setIsWorking(true)
    addLog('Started working - connecting to coordinator...')

    const wsProtocol = API_URL.startsWith('https') ? 'wss' : 'ws'
    const wsHost = API_URL.replace(/^https?:\/\//, '')
    const wsUrl = `${wsProtocol}://${wsHost}/ws/worker/${currentWorker.id}`
    const ws = new WebSocket(wsUrl)
    
    ws.onopen = () => {
      addLog('Connected to coordinator')
      ws.send(JSON.stringify({ type: 'request_task' }))
    }

    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'task_assigned') {
        const task = data.task as Task
        setCurrentTask(task)
        addLog(`Received task ${task.task_index} (payout: $${task.payout.toFixed(4)})`)
        
        const { result, flops } = performCompute(task)
        
        addLog(`Completed task - ${formatNumber(flops)} FLOPS`)
        
        ws.send(JSON.stringify({
          type: 'task_complete',
          task_id: task.id,
          result: result,
          flops: flops
        }))
        
        setCurrentTask(null)
        
        const workerRes = await fetch(`${API_URL}/workers/${currentWorker!.id}`)
        if (workerRes.ok) {
          setWorker(await workerRes.json())
        }
        
        setTimeout(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'request_task' }))
          }
        }, 500)
      } else if (data.type === 'no_tasks') {
        addLog('No tasks available - waiting...')
        setTimeout(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'request_task' }))
          }
        }, 2000)
      } else if (data.type === 'new_tasks_available') {
        addLog('New tasks available!')
        ws.send(JSON.stringify({ type: 'request_task' }))
      }
    }

    ws.onclose = () => {
      addLog('Disconnected from coordinator')
      setIsWorking(false)
    }

    ws.onerror = () => {
      addLog('Connection error')
    }

    wsRef.current = ws
  }

  const stopWorking = () => {
    if (wsRef.current) {
      wsRef.current.close()
    }
    if (workIntervalRef.current) {
      clearInterval(workIntervalRef.current)
    }
    setIsWorking(false)
    setCurrentTask(null)
    addLog('Stopped working')
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Cpu className="h-4 w-4" />
              Tasks Completed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{worker?.total_tasks_completed || 0}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              FLOPS Contributed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(worker?.total_flops_contributed || 0)}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Wallet className="h-4 w-4" />
              Total Earnings
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">${(worker?.total_earnings || 0).toFixed(4)}</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Server className="h-5 w-5" />
            Worker Node
          </CardTitle>
          <CardDescription>
            Contribute your device's compute power to the network
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!worker && (
            <div className="space-y-2">
              <Label>Worker Name</Label>
              <Input 
                value={workerName} 
                onChange={(e) => setWorkerName(e.target.value)}
                placeholder="Enter worker name"
              />
            </div>
          )}
          
          <div className="flex gap-2">
            {!isWorking ? (
              <Button onClick={startWorking} className="flex-1">
                <Play className="h-4 w-4 mr-2" />
                Start Contributing
              </Button>
            ) : (
              <Button onClick={stopWorking} variant="destructive" className="flex-1">
                <Square className="h-4 w-4 mr-2" />
                Stop
              </Button>
            )}
          </div>

          {currentTask && (
            <div className="bg-muted p-4 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="h-4 w-4 animate-pulse text-green-500" />
                <span className="font-medium">Processing Task {currentTask.task_index}</span>
              </div>
              <Progress value={50} className="animate-pulse" />
            </div>
          )}

          <div className="bg-black rounded-lg p-4 h-48 overflow-y-auto font-mono text-xs text-green-400">
            {taskLog.length === 0 ? (
              <p className="text-gray-500">Worker logs will appear here...</p>
            ) : (
              taskLog.map((log, i) => (
                <div key={i}>{log}</div>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function MilestonesDashboard() {
  const [stats, setStats] = useState<NetworkStats | null>(null)
  const [activityData, setActivityData] = useState<Array<{ time: string, tasks: number, gflops: number }>>([])

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch(`${API_URL}/stats`)
        const data = await res.json()
        setStats(data)
        
        setActivityData(prev => {
          const newPoint = {
            time: new Date().toLocaleTimeString(),
            tasks: data.total_tasks_completed,
            gflops: data.total_gflops
          }
          return [...prev.slice(-20), newPoint]
        })
      } catch (err) {
        console.error('Failed to fetch stats:', err)
      }
    }

    fetchStats()
    const interval = setInterval(fetchStats, 3000)
    return () => clearInterval(interval)
  }, [])

  if (!stats) {
    return <div className="flex items-center justify-center h-64">Loading...</div>
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Users className="h-4 w-4" />
              Active Devices
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total_devices}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Total GFLOPS
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(stats.total_gflops)}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Jobs Completed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total_jobs_completed}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <DollarSign className="h-4 w-4" />
              Total Payouts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">${stats.total_payments.toFixed(2)}</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Network Milestones
          </CardTitle>
          <CardDescription>
            Progress toward global compute capacity goals
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {stats.milestones.map((milestone, i) => (
            <div key={i} className="space-y-2">
              <div className="flex justify-between items-center">
                <div>
                  <h4 className="font-medium">{milestone.name}</h4>
                  <p className="text-sm text-muted-foreground">{milestone.capability}</p>
                </div>
                <Badge variant={stats.total_devices >= milestone.target_devices ? 'default' : 'outline'}>
                  {formatNumber(milestone.target_devices)} devices
                </Badge>
              </div>
              <Progress 
                value={Math.min((stats.total_devices / milestone.target_devices) * 100, 100)} 
                className="h-2"
              />
              <p className="text-xs text-muted-foreground text-right">
                {((stats.total_devices / milestone.target_devices) * 100).toFixed(6)}% complete
              </p>
            </div>
          ))}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Network Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={activityData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Line type="monotone" dataKey="tasks" stroke="#8884d8" name="Tasks" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Active Workers</CardTitle>
          </CardHeader>
          <CardContent>
            {stats.active_workers.length === 0 ? (
              <p className="text-muted-foreground text-center py-8">No active workers</p>
            ) : (
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {stats.active_workers.map((w) => (
                  <div key={w.id} className="flex justify-between items-center p-2 bg-muted rounded">
                    <span className="font-medium">{w.name}</span>
                    <div className="text-sm text-muted-foreground">
                      {w.total_tasks_completed} tasks | ${w.total_earnings.toFixed(4)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Payments</CardTitle>
        </CardHeader>
        <CardContent>
          {stats.recent_payments.length === 0 ? (
            <p className="text-muted-foreground text-center py-4">No payments yet</p>
          ) : (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {stats.recent_payments.slice().reverse().map((p) => (
                <div key={p.id} className="flex justify-between items-center p-2 border-b">
                  <span className="text-sm">Worker {p.worker_id.slice(0, 8)}...</span>
                  <div className="flex items-center gap-4">
                    <span className="text-green-600 font-medium">${p.amount.toFixed(4)}</span>
                    <span className="text-xs text-muted-foreground">
                      {new Date(p.timestamp * 1000).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            Oxygen<span className="text-purple-400">TM</span>
          </h1>
          <p className="text-slate-300">Distributed Compute Network for AI</p>
          <p className="text-sm text-slate-400 mt-1">
            Leveraging idle device compute to build a global AI infrastructure
          </p>
        </div>

        <Tabs defaultValue="customer" className="w-full">
          <TabsList className="grid w-full grid-cols-3 mb-6">
            <TabsTrigger value="customer">Customer Dashboard</TabsTrigger>
            <TabsTrigger value="worker">Worker Node</TabsTrigger>
            <TabsTrigger value="milestones">Network Stats</TabsTrigger>
          </TabsList>
          
          <TabsContent value="customer">
            <CustomerDashboard />
          </TabsContent>
          
          <TabsContent value="worker">
            <WorkerPortal />
          </TabsContent>
          
          <TabsContent value="milestones">
            <MilestonesDashboard />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

export default App
