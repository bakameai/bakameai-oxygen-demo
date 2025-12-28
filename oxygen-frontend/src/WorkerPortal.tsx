import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Cpu, DollarSign, CheckCircle, Play, Square, ExternalLink, Loader2 } from 'lucide-react'
import * as tf from '@tensorflow/tfjs'
import * as mobilenet from '@tensorflow-models/mobilenet'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

type KernelType = 'image_blur' | 'image_edge' | 'image_grayscale' | 'file_hash' | 'text_wordcount' | 'matrix_multiply' | 'image_classify' | 'model_finetune' | 'text_embed' | 'image_embed' | 'video_analyze'

// Global MobileNet model (loaded once, reused for all classifications)
let mobilenetModel: mobilenet.MobileNet | null = null

interface Worker {
  id: string
  name: string
  total_tasks_completed: number
  total_compute_time_ms: number
  total_earnings: number
  verified_tasks: number
}

interface Task {
  id: string
  job_id: string
  task_index: number
  shard_id: string
  shard_params: Record<string, unknown>
  payout: number
}

function WorkerPortal() {
  const [worker, setWorker] = useState<Worker | null>(null)
  const [workerName, setWorkerName] = useState(`Worker-${Math.random().toString(36).slice(2, 8)}`)
  const [isWorking, setIsWorking] = useState(false)
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [processingStatus, setProcessingStatus] = useState<string>('')
  const wsRef = useRef<WebSocket | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => [...prev.slice(-50), `[${timestamp}] ${message}`])
  }

  const registerWorker = async () => {
    try {
      const deviceInfo = {
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        cores: navigator.hardwareConcurrency || 4,
        memory: (navigator as unknown as { deviceMemory?: number }).deviceMemory || 'unknown'
      }
      
            // Report worker capabilities for capability-based routing
            const capabilities = {
              supported_kernels: ['image_blur', 'image_edge', 'image_grayscale', 'file_hash', 'text_wordcount', 'matrix_multiply', 'image_classify', 'model_finetune', 'text_embed', 'image_embed', 'video_analyze'],
              has_gpu: false, // Browser-based workers don't have direct GPU access
              ram_gb: (navigator as unknown as { deviceMemory?: number }).deviceMemory || 4,
              max_concurrent_tasks: 1
            }
      
      const res = await fetch(`${API_URL}/workers/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: workerName, device_info: deviceInfo, capabilities })
      })
      
      if (res.ok) {
        const data = await res.json()
        setWorker(data)
        addLog(`Registered as worker: ${data.id.slice(0, 8)}...`)
        return data
      }
    } catch (err) {
      addLog('Failed to register worker')
    }
    return null
  }

  const applyImageFilter = async (
    imageData: ImageData, 
    filterType: string
  ): Promise<ImageData> => {
    const data = imageData.data
    const width = imageData.width
    const height = imageData.height
    const output = new Uint8ClampedArray(data)

    if (filterType === 'image_blur') {
      // Gaussian blur (3x3 kernel)
      const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1]
      const kernelSum = 16
      
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          for (let c = 0; c < 3; c++) {
            let sum = 0
            for (let ky = -1; ky <= 1; ky++) {
              for (let kx = -1; kx <= 1; kx++) {
                const idx = ((y + ky) * width + (x + kx)) * 4 + c
                sum += data[idx] * kernel[(ky + 1) * 3 + (kx + 1)]
              }
            }
            output[(y * width + x) * 4 + c] = sum / kernelSum
          }
        }
      }
    } else if (filterType === 'image_edge') {
      // Sobel edge detection
      const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
      const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
      
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          let gx = 0, gy = 0
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const idx = ((y + ky) * width + (x + kx)) * 4
              const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3
              gx += gray * sobelX[(ky + 1) * 3 + (kx + 1)]
              gy += gray * sobelY[(ky + 1) * 3 + (kx + 1)]
            }
          }
          const magnitude = Math.min(255, Math.sqrt(gx * gx + gy * gy))
          const outIdx = (y * width + x) * 4
          output[outIdx] = magnitude
          output[outIdx + 1] = magnitude
          output[outIdx + 2] = magnitude
        }
      }
    } else if (filterType === 'image_grayscale') {
      for (let i = 0; i < data.length; i += 4) {
        const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114
        output[i] = gray
        output[i + 1] = gray
        output[i + 2] = gray
      }
    }

    return new ImageData(output, width, height)
  }

  // Process image tile task
  const processImageTask = async (task: Task, kernelType: KernelType, workerId: string) => {
    const startTime = performance.now()
    setProcessingStatus('Downloading source image...')
    
    try {
      const imgRes = await fetch(`${API_URL}/jobs/${task.job_id}/source-image`)
      const imgBlob = await imgRes.blob()
      
      const img = new Image()
      img.crossOrigin = 'anonymous'
      
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve()
        img.onerror = reject
        img.src = URL.createObjectURL(imgBlob)
      })
      
      setProcessingStatus('Extracting tile...')
      
      const row = task.shard_params?.row as number ?? 0
      const col = task.shard_params?.col as number ?? 0
      const gridSize = task.shard_params?.grid_size as number ?? 4
      
      const tileWidth = Math.floor(img.width / gridSize)
      const tileHeight = Math.floor(img.height / gridSize)
      const startX = col * tileWidth
      const startY = row * tileHeight
      
      const canvas = canvasRef.current!
      canvas.width = tileWidth
      canvas.height = tileHeight
      const ctx = canvas.getContext('2d')!
      
      ctx.drawImage(img, startX, startY, tileWidth, tileHeight, 0, 0, tileWidth, tileHeight)
      
      setProcessingStatus(`Applying ${kernelType.replace('image_', '')} filter...`)
      
      const imageData = ctx.getImageData(0, 0, tileWidth, tileHeight)
      const processedData = await applyImageFilter(imageData, kernelType)
      ctx.putImageData(processedData, 0, 0)
      
      setProcessingStatus('Uploading result...')
      
      const resultBlob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => resolve(blob!), 'image/png')
      })
      
      const arrayBuffer = await resultBlob.arrayBuffer()
      const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer)
      const hashArray = Array.from(new Uint8Array(hashBuffer))
      const resultHash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
      
      const computeTime = performance.now() - startTime
      
      const formData = new FormData()
      formData.append('worker_id', workerId)
      formData.append('tile_row', row.toString())
      formData.append('tile_col', col.toString())
      formData.append('compute_time_ms', computeTime.toString())
      formData.append('result_hash', resultHash)
      formData.append('file', resultBlob, 'tile.png')
      
      const submitRes = await fetch(`${API_URL}/jobs/${task.job_id}/submit-tile`, {
        method: 'POST',
        body: formData
      })
      
      if (submitRes.ok) {
        const result = await submitRes.json()
        addLog(`Completed tile (${row},${col}) - ${computeTime.toFixed(0)}ms - ${result.verified ? 'VERIFIED' : 'submitted'} - earned $${task.payout.toFixed(4)}`)
        await refreshWorkerStats()
      }
      return true
    } catch (err) {
      addLog(`Error processing image tile: ${err}`)
      return false
    }
  }

  // Process file hash task
  const processFileHashTask = async (task: Task, dataId: string, workerId: string) => {
    const startTime = performance.now()
    setProcessingStatus('Downloading chunk...')
    
    try {
      const chunkStart = task.shard_params?.start as number ?? 0
      const chunkEnd = task.shard_params?.end as number ?? 0
      
      const res = await fetch(`${API_URL}/data/${dataId}/range?start=${chunkStart}&end=${chunkEnd}`)
      const chunkData = await res.arrayBuffer()
      
      setProcessingStatus('Computing SHA-256 hash...')
      
      const hashBuffer = await crypto.subtle.digest('SHA-256', chunkData)
      const hashArray = Array.from(new Uint8Array(hashBuffer))
      const resultHash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
      
      const computeTime = performance.now() - startTime
      
      setProcessingStatus('Submitting result...')
      
      const formData = new FormData()
      formData.append('worker_id', workerId)
      formData.append('compute_time_ms', computeTime.toString())
      formData.append('result_hash', resultHash)
      formData.append('result_data', JSON.stringify({ chunk_hash: resultHash }))
      
      const submitRes = await fetch(`${API_URL}/tasks/${task.id}/submit`, {
        method: 'POST',
        body: formData
      })
      
      if (submitRes.ok) {
        const result = await submitRes.json()
        addLog(`Hashed chunk ${task.shard_id} - ${computeTime.toFixed(0)}ms - ${result.awaiting_validation ? 'awaiting validation' : 'submitted'}`)
        await refreshWorkerStats()
      }
      return true
    } catch (err) {
      addLog(`Error processing file hash: ${err}`)
      return false
    }
  }

  // Process text word count task
  const processTextWordCountTask = async (task: Task, dataId: string, workerId: string) => {
    const startTime = performance.now()
    setProcessingStatus('Downloading text chunk...')
    
    try {
      const chunkStart = task.shard_params?.start as number ?? 0
      const chunkEnd = task.shard_params?.end as number ?? 0
      
      const res = await fetch(`${API_URL}/data/${dataId}/range?start=${chunkStart}&end=${chunkEnd}`)
      const textData = await res.text()
      
      setProcessingStatus('Counting words...')
      
      const words = textData.toLowerCase().match(/\b[a-z]+\b/g) || []
      const wordCounts: Record<string, number> = {}
      for (const word of words) {
        wordCounts[word] = (wordCounts[word] || 0) + 1
      }
      
      const resultHash = await computeHash(JSON.stringify(wordCounts))
      const computeTime = performance.now() - startTime
      
      setProcessingStatus('Submitting result...')
      
      const formData = new FormData()
      formData.append('worker_id', workerId)
      formData.append('compute_time_ms', computeTime.toString())
      formData.append('result_hash', resultHash)
      formData.append('result_data', JSON.stringify({ word_counts: wordCounts, total_words: words.length, lines_processed: textData.split('\n').length }))
      
      const submitRes = await fetch(`${API_URL}/tasks/${task.id}/submit`, {
        method: 'POST',
        body: formData
      })
      
      if (submitRes.ok) {
        addLog(`Counted ${words.length} words in ${task.shard_id} - ${computeTime.toFixed(0)}ms`)
        await refreshWorkerStats()
      }
      return true
    } catch (err) {
      addLog(`Error processing word count: ${err}`)
      return false
    }
  }

  // Process matrix multiplication task
  const processMatrixMultiplyTask = async (task: Task, jobParams: Record<string, unknown>, workerId: string) => {
    const startTime = performance.now()
    setProcessingStatus('Computing matrix block...')
    
    try {
      const blockRow = task.shard_params?.block_row as number ?? 0
      const blockCol = task.shard_params?.block_col as number ?? 0
      const blockK = task.shard_params?.block_k as number ?? 0
      const blockSize = jobParams?.block_size as number ?? 16
      // matrixSize is available in jobParams but not needed for block computation
      
      // Generate deterministic random matrices based on seed
      const seed = blockRow * 1000 + blockCol * 100 + blockK
      const blockA = generateDeterministicBlock(seed, blockSize, 'A')
      const blockB = generateDeterministicBlock(seed, blockSize, 'B')
      
      // Compute block multiplication
      const resultBlock = multiplyBlocks(blockA, blockB, blockSize)
      
      // Compute checksum
      let checksum = 0
      for (let i = 0; i < blockSize; i++) {
        for (let j = 0; j < blockSize; j++) {
          checksum += resultBlock[i * blockSize + j]
        }
      }
      
      const resultHash = await computeHash(`${blockRow}-${blockCol}-${blockK}-${checksum}`)
      const computeTime = performance.now() - startTime
      
      setProcessingStatus('Submitting result...')
      
      const formData = new FormData()
      formData.append('worker_id', workerId)
      formData.append('compute_time_ms', computeTime.toString())
      formData.append('result_hash', resultHash)
      formData.append('result_data', JSON.stringify({ block_checksum: checksum, block_row: blockRow, block_col: blockCol, block_k: blockK }))
      
      const submitRes = await fetch(`${API_URL}/tasks/${task.id}/submit`, {
        method: 'POST',
        body: formData
      })
      
      if (submitRes.ok) {
        addLog(`Computed matrix block (${blockRow},${blockCol},${blockK}) - ${computeTime.toFixed(0)}ms`)
        await refreshWorkerStats()
      }
      return true
    } catch (err) {
      addLog(`Error processing matrix block: ${err}`)
      return false
    }
  }

  // Process image classification task using MobileNet
  const processImageClassifyTask = async (task: Task, workerId: string) => {
    const startTime = performance.now()
    setProcessingStatus('Loading AI model...')
    
    try {
      // Load MobileNet model if not already loaded
      if (!mobilenetModel) {
        addLog('Loading MobileNet AI model (first time only)...')
        await tf.ready()
        mobilenetModel = await mobilenet.load()
        addLog('MobileNet model loaded successfully')
      }
      
      setProcessingStatus('Downloading image...')
      
      // Download the image from the backend
      const imgRes = await fetch(`${API_URL}/jobs/${task.job_id}/source-image`)
      const imgBlob = await imgRes.blob()
      
      // Create an image element
      const img = document.createElement('img')
      img.crossOrigin = 'anonymous'
      
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve()
        img.onerror = reject
        img.src = URL.createObjectURL(imgBlob)
      })
      
      setProcessingStatus('Running AI classification...')
      
      // Run MobileNet classification
      const predictions = await mobilenetModel.classify(img)
      
      // Get top prediction
      const topPrediction = predictions[0]
      const topClass = topPrediction?.className || 'unknown'
      const confidence = topPrediction?.probability || 0
      
      // Format predictions for result
      const formattedPredictions = predictions.map(p => ({
        className: p.className,
        probability: p.probability
      }))
      
      // Compute result hash based on classification
      const resultHash = await computeHash(JSON.stringify(formattedPredictions))
      const computeTime = performance.now() - startTime
      
      setProcessingStatus('Submitting result...')
      
      const formData = new FormData()
      formData.append('worker_id', workerId)
      formData.append('compute_time_ms', computeTime.toString())
      formData.append('result_hash', resultHash)
      formData.append('result_data', JSON.stringify({
        predictions: formattedPredictions,
        top_class: topClass,
        confidence: confidence
      }))
      
      const submitRes = await fetch(`${API_URL}/tasks/${task.id}/submit`, {
        method: 'POST',
        body: formData
      })
      
      if (submitRes.ok) {
        addLog(`Classified: "${topClass}" (${(confidence * 100).toFixed(1)}% confidence) - ${computeTime.toFixed(0)}ms`)
        await refreshWorkerStats()
      }
      
      // Clean up
      URL.revokeObjectURL(img.src)
      
      return true
    } catch (err) {
      addLog(`Error classifying image: ${err}`)
      return false
    }
  }

    // Process model fine-tuning task
    const processModelFinetuneTask = async (task: Task, dataId: string, _jobParams: Record<string, unknown>, workerId: string) => {
      const startTime = performance.now()
      setProcessingStatus('Loading training data...')
    
      try {
        // Get training data from backend
        const dataRes = await fetch(`${API_URL}/data/${dataId}`)
        if (!dataRes.ok) throw new Error('Failed to fetch training data')
        const trainingDataJson = await dataRes.text()
        const trainingData = JSON.parse(trainingDataJson) as { images: string[], labels: string[] }
      
        const roundIndex = task.shard_params?.round_index as number || 0
        const epochs = task.shard_params?.epochs as number || 1
        const totalRounds = task.shard_params?.total_rounds as number || 5
      
        addLog(`Training round ${roundIndex + 1}/${totalRounds} (${epochs} epochs)`)
        setProcessingStatus(`Training round ${roundIndex + 1}/${totalRounds}...`)
      
        // Load MobileNet model if not already loaded
        if (!mobilenetModel) {
          addLog('Loading MobileNet AI model...')
          await tf.ready()
          mobilenetModel = await mobilenet.load()
          addLog('MobileNet model loaded')
        }
      
        // Get unique labels
        const uniqueLabels = [...new Set(trainingData.labels)]
        const numClasses = uniqueLabels.length
      
        // Create a simple transfer learning model on top of MobileNet
        // We'll use the MobileNet features and add a small classifier
        setProcessingStatus('Preparing training images...')
      
        // Process training images
        const imagePromises = trainingData.images.map(async (base64Img) => {
          const img = document.createElement('img')
          await new Promise<void>((resolve, reject) => {
            img.onload = () => resolve()
            img.onerror = reject
            img.src = base64Img
          })
        
          // Resize to MobileNet input size (224x224)
          const canvas = document.createElement('canvas')
          canvas.width = 224
          canvas.height = 224
          const ctx = canvas.getContext('2d')!
          ctx.drawImage(img, 0, 0, 224, 224)
        
          // Convert to tensor
          return tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0)
        })
      
        const imageTensors = await Promise.all(imagePromises)
      
        // Create labels tensor (one-hot encoded)
        const labelIndices = trainingData.labels.map(l => uniqueLabels.indexOf(l))
        const labelsTensor = tf.oneHot(tf.tensor1d(labelIndices, 'int32'), numClasses)
      
        // Concatenate all images
        const xs = tf.concat(imageTensors, 0)
      
        // Create a simple model for fine-tuning
        // In a real scenario, we'd use MobileNet's feature extraction
        // For demo, we'll create a small classifier
        const model = tf.sequential({
          layers: [
            tf.layers.flatten({ inputShape: [224, 224, 3] }),
            tf.layers.dense({ units: 128, activation: 'relu' }),
            tf.layers.dropout({ rate: 0.2 }),
            tf.layers.dense({ units: numClasses, activation: 'softmax' })
          ]
        })
      
        model.compile({
          optimizer: tf.train.adam(0.001),
          loss: 'categoricalCrossentropy',
          metrics: ['accuracy']
        })
      
        // Get initial loss
        const initialEval = model.evaluate(xs, labelsTensor) as tf.Tensor[]
        const lossBefore = (await initialEval[0].data())[0]
      
        setProcessingStatus(`Training (${epochs} epochs)...`)
      
        // Train the model
        const history = await model.fit(xs, labelsTensor, {
          epochs: epochs,
          batchSize: Math.min(4, trainingData.images.length),
          verbose: 0
        })
      
        // Get final metrics
        const finalEval = model.evaluate(xs, labelsTensor) as tf.Tensor[]
        const lossAfter = (await finalEval[0].data())[0]
        const accuracy = (await finalEval[1].data())[0]
      
        // Clean up tensors
        xs.dispose()
        labelsTensor.dispose()
        imageTensors.forEach(t => t.dispose())
        initialEval.forEach(t => t.dispose())
        finalEval.forEach(t => t.dispose())
        model.dispose()
      
        const computeTime = performance.now() - startTime
      
        // Create result with training metrics
        const resultData = {
          round_index: roundIndex,
          epochs: epochs,
          loss_before: lossBefore,
          loss_after: lossAfter,
          accuracy: accuracy,
          samples_trained: trainingData.images.length,
          training_history: history.history
        }
      
        const resultHash = await computeHash(JSON.stringify(resultData))
      
        setProcessingStatus('Submitting training results...')
      
        const formData = new FormData()
        formData.append('worker_id', workerId)
        formData.append('compute_time_ms', computeTime.toString())
        formData.append('result_hash', resultHash)
        formData.append('result_data', JSON.stringify(resultData))
      
        const submitRes = await fetch(`${API_URL}/tasks/${task.id}/submit`, {
          method: 'POST',
          body: formData
        })
      
        if (submitRes.ok) {
          addLog(`Round ${roundIndex + 1}: Loss ${lossBefore.toFixed(4)} → ${lossAfter.toFixed(4)}, Acc: ${(accuracy * 100).toFixed(1)}% - ${computeTime.toFixed(0)}ms`)
          await refreshWorkerStats()
        }
      
        return true
      } catch (err) {
        addLog(`Error in fine-tuning: ${err}`)
        return false
      }
    }

  // Process text embedding task - generate embeddings for text items
  const processTextEmbedTask = async (task: Task, dataId: string, workerId: string) => {
    const startTime = performance.now()
    setProcessingStatus('Downloading text items...')
    
    try {
      // Get the text data
      const res = await fetch(`${API_URL}/data/${dataId}`)
      const textData = await res.text()
      
      // Parse items (lines or JSON array)
      let items: string[] = []
      try {
        const parsed = JSON.parse(textData)
        if (Array.isArray(parsed)) {
          items = parsed.map(item => typeof item === 'object' ? JSON.stringify(item) : String(item))
        } else {
          items = textData.trim().split('\n').filter(l => l.trim())
        }
      } catch {
        items = textData.trim().split('\n').filter(l => l.trim())
      }
      
      // Get batch range from task params
      const startIndex = task.shard_params?.start_index as number ?? 0
      const endIndex = task.shard_params?.end_index as number ?? items.length
      const batchItems = items.slice(startIndex, endIndex)
      
      setProcessingStatus(`Generating embeddings for ${batchItems.length} items...`)
      
      // Generate mock embeddings (in production, would use a real embedding model)
      // Each embedding is a 128-dimensional vector
      const embeddings: number[][] = []
      for (const item of batchItems) {
        // Generate deterministic embedding based on item content
        const embedding: number[] = []
        for (let i = 0; i < 128; i++) {
          // Simple hash-based embedding (mock)
          let hash = 0
          for (let j = 0; j < item.length; j++) {
            hash = ((hash << 5) - hash + item.charCodeAt(j) + i) | 0
          }
          embedding.push(Math.sin(hash) * 0.5 + 0.5)
        }
        embeddings.push(embedding)
      }
      
      const resultHash = await computeHash(JSON.stringify(embeddings))
      const computeTime = performance.now() - startTime
      
      setProcessingStatus('Submitting embeddings...')
      
      const formData = new FormData()
      formData.append('worker_id', workerId)
      formData.append('compute_time_ms', computeTime.toString())
      formData.append('result_hash', resultHash)
      formData.append('result_data', JSON.stringify({
        embeddings: embeddings,
        items: batchItems,
        batch_index: task.shard_params?.batch_index ?? 0
      }))
      
      const submitRes = await fetch(`${API_URL}/tasks/${task.id}/submit`, {
        method: 'POST',
        body: formData
      })
      
      if (submitRes.ok) {
        addLog(`Generated ${embeddings.length} text embeddings - ${computeTime.toFixed(0)}ms`)
        await refreshWorkerStats()
      }
      return true
    } catch (err) {
      addLog(`Error generating text embeddings: ${err}`)
      return false
    }
  }

  // Process image embedding task - generate embeddings for images
  const processImageEmbedTask = async (task: Task, dataId: string, workerId: string) => {
    const startTime = performance.now()
    setProcessingStatus('Downloading image collection...')
    
    try {
      // Get the image collection data
      const res = await fetch(`${API_URL}/data/${dataId}`)
      const jsonData = await res.json()
      
      let images: string[] = []
      if (Array.isArray(jsonData)) {
        images = jsonData
      } else if (jsonData.images) {
        images = jsonData.images
      }
      
      // Get batch range from task params
      const startIndex = task.shard_params?.start_index as number ?? 0
      const endIndex = task.shard_params?.end_index as number ?? images.length
      const batchImages = images.slice(startIndex, endIndex)
      
      setProcessingStatus(`Generating embeddings for ${batchImages.length} images...`)
      
      // Load MobileNet for feature extraction if not loaded
      if (!mobilenetModel) {
        addLog('Loading MobileNet for image embeddings...')
        await tf.ready()
        mobilenetModel = await mobilenet.load()
      }
      
      // Generate embeddings for each image
      const embeddings: number[][] = []
      const imageIds: string[] = []
      
      for (let i = 0; i < batchImages.length; i++) {
        const imgData = batchImages[i]
        imageIds.push(`img_${startIndex + i}`)
        
        // Create image element from base64
        const img = document.createElement('img')
        img.crossOrigin = 'anonymous'
        
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve()
          img.onerror = reject
          img.src = imgData
        })
        
        // Get embedding from MobileNet (use infer to get feature vector)
        const tensor = tf.browser.fromPixels(img)
        const resized = tf.image.resizeBilinear(tensor, [224, 224])
        const expanded = resized.expandDims(0)
        const normalized = expanded.div(255.0)
        
        // Get predictions as embedding (simplified - in production would use intermediate layer)
        const predictions = await mobilenetModel.classify(img)
        
        // Create embedding from predictions (mock - in production would use actual feature vector)
        const embedding: number[] = []
        for (let j = 0; j < 128; j++) {
          const pred = predictions[j % predictions.length]
          embedding.push(pred.probability * Math.sin(j) + 0.5)
        }
        embeddings.push(embedding)
        
        // Cleanup tensors
        tensor.dispose()
        resized.dispose()
        expanded.dispose()
        normalized.dispose()
      }
      
      const resultHash = await computeHash(JSON.stringify(embeddings))
      const computeTime = performance.now() - startTime
      
      setProcessingStatus('Submitting image embeddings...')
      
      const formData = new FormData()
      formData.append('worker_id', workerId)
      formData.append('compute_time_ms', computeTime.toString())
      formData.append('result_hash', resultHash)
      formData.append('result_data', JSON.stringify({
        embeddings: embeddings,
        image_ids: imageIds,
        batch_index: task.shard_params?.batch_index ?? 0
      }))
      
      const submitRes = await fetch(`${API_URL}/tasks/${task.id}/submit`, {
        method: 'POST',
        body: formData
      })
      
      if (submitRes.ok) {
        addLog(`Generated ${embeddings.length} image embeddings - ${computeTime.toFixed(0)}ms`)
        await refreshWorkerStats()
      }
      return true
    } catch (err) {
      addLog(`Error generating image embeddings: ${err}`)
      return false
    }
  }

  // Process video analysis task - detect objects in video frames
  const processVideoAnalyzeTask = async (task: Task, workerId: string) => {
    const startTime = performance.now()
    setProcessingStatus('Analyzing video frames...')
    
    try {
      // Get frame range from task params
      const startFrame = task.shard_params?.start_frame as number ?? 0
      const endFrame = task.shard_params?.end_frame as number ?? 10
      // totalFrames available in task.shard_params if needed for progress calculation
      
      setProcessingStatus(`Analyzing frames ${startFrame}-${endFrame}...`)
      
      // Load MobileNet for object detection if not loaded
      if (!mobilenetModel) {
        addLog('Loading MobileNet for video analysis...')
        await tf.ready()
        mobilenetModel = await mobilenet.load()
      }
      
      // Generate mock detections for each frame
      // In production, would extract actual frames from video and run detection
      const detections: Array<{
        frame: number
        timestamp: number
        label: string
        confidence: number
        bbox: { x: number, y: number, width: number, height: number }
      }> = []
      
      const possibleLabels = ['person', 'car', 'dog', 'cat', 'bicycle', 'truck', 'bird', 'chair', 'table', 'phone']
      
      for (let frame = startFrame; frame < endFrame; frame++) {
        // Simulate frame analysis with deterministic random detections
        const seed = frame * 12345
        const numDetections = (seed % 3) + 1 // 1-3 detections per frame
        
        for (let d = 0; d < numDetections; d++) {
          const labelIndex = (seed + d * 7) % possibleLabels.length
          const confidence = 0.5 + ((seed + d * 13) % 50) / 100 // 0.5-1.0
          
          detections.push({
            frame: frame,
            timestamp: frame / 30, // Assume 30fps
            label: possibleLabels[labelIndex],
            confidence: confidence,
            bbox: {
              x: (seed % 80) / 100,
              y: ((seed + d) % 80) / 100,
              width: 0.1 + (seed % 20) / 100,
              height: 0.1 + ((seed + d) % 20) / 100
            }
          })
        }
      }
      
      const resultHash = await computeHash(JSON.stringify(detections))
      const computeTime = performance.now() - startTime
      
      setProcessingStatus('Submitting detection results...')
      
      const formData = new FormData()
      formData.append('worker_id', workerId)
      formData.append('compute_time_ms', computeTime.toString())
      formData.append('result_hash', resultHash)
      formData.append('result_data', JSON.stringify({
        detections: detections,
        frames_analyzed: endFrame - startFrame,
        batch_index: task.shard_params?.batch_index ?? 0
      }))
      
      const submitRes = await fetch(`${API_URL}/tasks/${task.id}/submit`, {
        method: 'POST',
        body: formData
      })
      
      if (submitRes.ok) {
        addLog(`Analyzed frames ${startFrame}-${endFrame}, found ${detections.length} objects - ${computeTime.toFixed(0)}ms`)
        await refreshWorkerStats()
      }
      return true
    } catch (err) {
      addLog(`Error analyzing video frames: ${err}`)
      return false
    }
  }

    // Helper: compute SHA-256 hash
    const computeHash = async (data: string): Promise<string> => {
    const encoder = new TextEncoder()
    const hashBuffer = await crypto.subtle.digest('SHA-256', encoder.encode(data))
    const hashArray = Array.from(new Uint8Array(hashBuffer))
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
  }

  // Helper: generate deterministic block for matrix
  const generateDeterministicBlock = (seed: number, size: number, _matrix: string): Float64Array => {
    const block = new Float64Array(size * size)
    let s = seed
    for (let i = 0; i < size * size; i++) {
      s = (s * 1103515245 + 12345) & 0x7fffffff
      block[i] = (s % 100) / 100.0
    }
    return block
  }

  // Helper: multiply two blocks
  const multiplyBlocks = (a: Float64Array, b: Float64Array, size: number): Float64Array => {
    const result = new Float64Array(size * size)
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        let sum = 0
        for (let k = 0; k < size; k++) {
          sum += a[i * size + k] * b[k * size + j]
        }
        result[i * size + j] = sum
      }
    }
    return result
  }

  // Helper: refresh worker stats
  const refreshWorkerStats = async () => {
    try {
      const workerRes = await fetch(`${API_URL}/workers/${worker!.id}`)
      if (workerRes.ok) {
        setWorker(await workerRes.json())
      }
    } catch {}
  }

  // Main task dispatcher
  const processTask = async (task: Task, kernelType: KernelType, dataId: string, jobParams: Record<string, unknown>, workerId: string) => {
    setProcessingStatus('Processing task...')
    
    try {
      let success = false
      
            if (kernelType === 'model_finetune') {
              // Distributed model fine-tuning
              success = await processModelFinetuneTask(task, dataId, jobParams, workerId)
            } else if (kernelType === 'image_classify') {
              // AI image classification using MobileNet
              success = await processImageClassifyTask(task, workerId)
            } else if (kernelType === 'text_embed') {
              // Text embeddings for semantic search
              success = await processTextEmbedTask(task, dataId, workerId)
            } else if (kernelType === 'image_embed') {
              // Image embeddings for visual search
              success = await processImageEmbedTask(task, dataId, workerId)
            } else if (kernelType === 'video_analyze') {
              // Video frame analysis for object detection
              success = await processVideoAnalyzeTask(task, workerId)
            } else if (kernelType.startsWith('image_')) {
              // Image processing filters (blur, edge, grayscale)
              success = await processImageTask(task, kernelType, workerId)
            } else if (kernelType === 'file_hash') {
              success = await processFileHashTask(task, dataId, workerId)
            } else if (kernelType === 'text_wordcount') {
              success = await processTextWordCountTask(task, dataId, workerId)
            } else if (kernelType === 'matrix_multiply') {
              success = await processMatrixMultiplyTask(task, jobParams, workerId)
            } else {
              addLog(`Unknown kernel type: ${kernelType}`)
            }
      
      setProcessingStatus('')
      return success
    } catch (err) {
      addLog(`Error processing task: ${err}`)
      setProcessingStatus('')
      return false
    }
  }

  const startWorking = async () => {
    let currentWorker = worker
    if (!currentWorker) {
      currentWorker = await registerWorker()
      if (!currentWorker) return
    }

    setIsWorking(true)
    addLog('Connecting to coordinator...')

    const wsProtocol = API_URL.startsWith('https') ? 'wss' : 'ws'
    const wsHost = API_URL.replace(/^https?:\/\//, '')
    const wsUrl = `${wsProtocol}://${wsHost}/ws/worker/${currentWorker.id}`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws
    
    ws.onopen = () => {
      addLog('Connected - requesting tasks...')
      ws.send(JSON.stringify({ type: 'request_task' }))
    }

    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'task_assignment') {
        if (data.task) {
          const task = data.task as Task
          const fragment = data.fragment as { shard_id: string, shard_params: Record<string, unknown> }
          const kernelType = data.kernel_type as KernelType
          const dataId = data.data_id as string
          const jobParams = data.job_params as Record<string, unknown> || {}
          const leaseSeconds = data.lease_seconds as number || 60
          
          // Update task with fragment info for processing
          task.shard_id = fragment.shard_id
          task.shard_params = fragment.shard_params
          
          setCurrentTask(task)
          addLog(`Received task ${fragment.shard_id} (${kernelType}) - lease: ${leaseSeconds}s`)
          
          await processTask(task, kernelType, dataId, jobParams, currentWorker.id)
          
          setCurrentTask(null)
          
          // Request next task
          setTimeout(() => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: 'request_task' }))
            }
          }, 500)
        } else {
          // No task available
          addLog(data.message || 'No tasks available - waiting...')
          setTimeout(() => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: 'request_task' }))
            }
          }, 3000)
        }
      } else if (data.type === 'new_job') {
        addLog(`New job available: ${data.fragment_count} fragments x ${data.tasks_per_fragment} workers`)
        ws.send(JSON.stringify({ type: 'request_task' }))
      } else if (data.type === 'heartbeat_ack') {
        // Heartbeat acknowledged
      }
    }

    ws.onclose = () => {
      addLog('Disconnected from coordinator')
      setIsWorking(false)
    }

    ws.onerror = () => {
      addLog('Connection error')
    }
  }

  const stopWorking = () => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsWorking(false)
    setCurrentTask(null)
    setProcessingStatus('')
    addLog('Stopped working')
  }

  // Refresh worker stats periodically
  useEffect(() => {
    if (!worker) return
    
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/workers/${worker.id}`)
        if (res.ok) {
          setWorker(await res.json())
        }
      } catch {}
    }, 5000)
    
    return () => clearInterval(interval)
  }, [worker])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-green-900 to-slate-900">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            Oxygen<span className="text-green-400">TM</span>
          </h1>
          <p className="text-slate-300 text-lg">Worker Portal</p>
          <p className="text-sm text-slate-400 mt-1">
            Contribute your device's compute power and earn rewards
          </p>
          <a 
            href="/" 
            className="inline-flex items-center gap-1 text-green-400 hover:text-green-300 text-sm mt-2"
          >
            Switch to Customer Portal <ExternalLink className="h-3 w-3" />
          </a>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <Card className="bg-white/95">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                Tasks Completed
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{worker?.total_tasks_completed || 0}</div>
              <p className="text-xs text-gray-500">
                {worker?.verified_tasks || 0} verified
              </p>
            </CardContent>
          </Card>
          
          <Card className="bg-white/95">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <CheckCircle className="h-4 w-4" />
                Compute Time
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {((worker?.total_compute_time_ms || 0) / 1000).toFixed(1)}s
              </div>
              <p className="text-xs text-gray-500">total processing</p>
            </CardContent>
          </Card>
          
          <Card className="bg-white/95">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <DollarSign className="h-4 w-4" />
                Total Earnings
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                ${(worker?.total_earnings || 0).toFixed(4)}
              </div>
              <p className="text-xs text-gray-500">$0.0001 per task</p>
            </CardContent>
          </Card>
        </div>

        {/* Worker Control */}
        <Card className="bg-white/95 mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Worker Node
            </CardTitle>
            <CardDescription>
              Process compute tasks for the distributed network
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {!isWorking && (
              <div className="space-y-2">
                <Label>Worker Name</Label>
                <Input 
                  value={workerName}
                  onChange={(e) => setWorkerName(e.target.value)}
                  placeholder="Enter worker name"
                />
              </div>
            )}
            
            <Button 
              onClick={isWorking ? stopWorking : startWorking}
              className={`w-full ${isWorking ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}`}
            >
              {isWorking ? (
                <>
                  <Square className="h-4 w-4 mr-2" />
                  Stop Contributing
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Start Contributing
                </>
              )}
            </Button>

            {/* Current Task */}
            {currentTask && (
              <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                <div className="flex items-center gap-2 mb-2">
                  <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                  <span className="font-medium text-blue-800">Processing Task</span>
                </div>
                <p className="text-sm text-blue-700">
                  Shard: {currentTask.shard_id}
                </p>
                {processingStatus && (
                  <p className="text-xs text-blue-600 mt-1">{processingStatus}</p>
                )}
              </div>
            )}

            {/* Hidden canvas for processing */}
            <canvas ref={canvasRef} className="hidden" />

            {/* Logs */}
            <div className="bg-gray-900 rounded-lg p-3 font-mono text-xs text-green-400 h-48 overflow-y-auto">
              {logs.length === 0 ? (
                <span className="text-gray-500">Worker logs will appear here...</span>
              ) : (
                logs.map((log, i) => (
                  <div key={i} className="py-0.5">{log}</div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        {/* How it works */}
        <Card className="bg-white/95">
          <CardHeader>
            <CardTitle>How It Works</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-gray-600 space-y-2">
            <p>1. Click "Start Contributing" to connect to the Oxygen network</p>
            <p>2. Your browser will receive compute tasks (image processing, file hashing, word counting, matrix math)</p>
            <p>3. Each task is processed using real compute kernels in your browser</p>
            <p>4. Results are verified with SHA-256 hashes and you earn $0.0001 per task</p>
            <p>5. The more tasks you process, the more you earn!</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default WorkerPortal
