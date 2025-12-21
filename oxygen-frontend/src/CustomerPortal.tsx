import { useState, useEffect, useRef, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Upload, Image, CheckCircle, Clock, Loader2, ExternalLink, FileText, Hash, Grid3X3, File } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

type KernelType = 'image_blur' | 'image_edge' | 'image_grayscale' | 'file_hash' | 'text_wordcount' | 'matrix_multiply' | 'image_classify' | 'model_finetune'

interface Kernel {
  type: KernelType
  name: string
  description: string
  input_type: string
}

interface Fragment {
  id: string
  job_id: string
  fragment_index: number
  shard_id: string
  shard_params: Record<string, unknown>
  status: string
  submissions: string[]
  consensus_hash?: string
  verified_result?: Record<string, unknown>
}

interface Job {
  id: string
  kernel_type: KernelType
  status: string
  created_at: number
  completed_at?: number
  total_fragments: number
  completed_fragments: number
  verified_fragments: number
  data_id?: string
  params: Record<string, unknown>
  result?: Record<string, unknown>
  escrow_amount: number
  escrow_remaining?: number
  lifecycle?: Array<{timestamp: number, event: string, details: Record<string, unknown>}>
  fragments?: Fragment[]
  tasks?: Task[]
}

interface Task {
  id: string
  shard_id: string
  shard_params: Record<string, unknown>
  status: string
  verified: boolean
  compute_time_ms: number
  result_hash?: string
  result_data?: Record<string, unknown>
}

const KERNELS: Kernel[] = [
  { type: 'model_finetune', name: 'AI Model Fine-Tuning', description: 'Distributed fine-tuning of MobileNet on custom images (real training)', input_type: 'training' },
  { type: 'image_classify', name: 'AI Image Classification', description: 'Classify images using MobileNet AI model (real inference)', input_type: 'image' },
  { type: 'image_blur', name: 'Gaussian Blur', description: 'Apply Gaussian blur filter to images', input_type: 'image' },
  { type: 'image_edge', name: 'Edge Detection', description: 'Sobel edge detection filter', input_type: 'image' },
  { type: 'image_grayscale', name: 'Grayscale', description: 'Convert image to grayscale', input_type: 'image' },
  { type: 'file_hash', name: 'File Hashing', description: 'Compute SHA-256 hashes of file chunks', input_type: 'any' },
  { type: 'text_wordcount', name: 'Word Count', description: 'Count word frequencies in text', input_type: 'text' },
  { type: 'matrix_multiply', name: 'Matrix Multiplication', description: 'Distributed matrix multiplication', input_type: 'matrix' },
]

function CustomerPortal() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [kernelType, setKernelType] = useState<KernelType>('image_blur')
  const [gridSize, setGridSize] = useState('4')
    const [matrixSize, setMatrixSize] = useState('64')
    const [blockSize, setBlockSize] = useState('16')
    const [textContent, setTextContent] = useState('')
    const [trainingImages, setTrainingImages] = useState<File[]>([])
    const [trainingLabels, setTrainingLabels] = useState<string[]>([])
    const [numRounds, setNumRounds] = useState('5')
    const [epochsPerRound, setEpochsPerRound] = useState('1')
  const [isUploading, setIsUploading] = useState(false)
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  
  const selectedKernel = KERNELS.find(k => k.type === kernelType)

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/jobs?customer_id=demo-customer`)
      const data = await res.json()
      setJobs(data)
    } catch (err) {
      console.error('Failed to fetch jobs:', err)
    }
  }, [])

  const fetchJobDetails = useCallback(async (jobId: string) => {
    try {
      const res = await fetch(`${API_URL}/jobs/${jobId}`)
      const data = await res.json()
      setSelectedJob(data)
    } catch (err) {
      console.error('Failed to fetch job details:', err)
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
        if (selectedJob && data.job_progress?.job_id === selectedJob.id) {
          fetchJobDetails(selectedJob.id)
        }
      }
    }
    
    wsRef.current = ws
    return () => ws.close()
  }, [fetchJobs, fetchJobDetails, selectedJob])

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        setSelectedFile(file)
        if (selectedKernel?.input_type === 'image') {
          const url = URL.createObjectURL(file)
          setPreviewUrl(url)
        } else if (selectedKernel?.input_type === 'text') {
          const reader = new FileReader()
          reader.onload = (e) => {
            setTextContent(e.target?.result as string || '')
          }
          reader.readAsText(file)
          setPreviewUrl(null)
        } else {
          setPreviewUrl(null)
        }
      }
    }

    const handleTrainingImageAdd = (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files
      if (files && files.length > 0) {
        const newImages = Array.from(files)
        setTrainingImages(prev => [...prev, ...newImages])
        // Initialize labels for new images
        setTrainingLabels(prev => [...prev, ...newImages.map(() => '')])
      }
    }

    const handleLabelChange = (index: number, label: string) => {
      setTrainingLabels(prev => {
        const newLabels = [...prev]
        newLabels[index] = label
        return newLabels
      })
    }

    const removeTrainingImage = (index: number) => {
      setTrainingImages(prev => prev.filter((_, i) => i !== index))
      setTrainingLabels(prev => prev.filter((_, i) => i !== index))
    }

  const handleSubmit = async () => {
    setIsUploading(true)
    try {
      let dataId: string | null = null
      
      // Upload data based on kernel type
      if (selectedKernel?.input_type === 'image' && selectedFile) {
        const formData = new FormData()
        formData.append('file', selectedFile)
        const uploadRes = await fetch(`${API_URL}/upload-image`, {
          method: 'POST',
          body: formData
        })
        if (!uploadRes.ok) throw new Error('Image upload failed')
        const result = await uploadRes.json()
        dataId = result.data_id
      } else if (selectedKernel?.input_type === 'any' && selectedFile) {
        const formData = new FormData()
        formData.append('file', selectedFile)
        const uploadRes = await fetch(`${API_URL}/upload`, {
          method: 'POST',
          body: formData
        })
        if (!uploadRes.ok) throw new Error('File upload failed')
        const result = await uploadRes.json()
        dataId = result.data_id
      } else if (selectedKernel?.input_type === 'text') {
        const content = textContent || (selectedFile ? await selectedFile.text() : '')
        if (!content) throw new Error('No text content provided')
        const blob = new Blob([content], { type: 'text/plain' })
        const formData = new FormData()
        formData.append('file', blob, 'text.txt')
        const uploadRes = await fetch(`${API_URL}/upload`, {
          method: 'POST',
          body: formData
        })
        if (!uploadRes.ok) throw new Error('Text upload failed')
        const result = await uploadRes.json()
        dataId = result.data_id
            } else if (selectedKernel?.input_type === 'matrix') {
              dataId = `matrix-${Date.now()}`
            } else if (selectedKernel?.input_type === 'training') {
              // Upload training images and labels as a bundle
              if (trainingImages.length === 0) throw new Error('No training images provided')
              if (trainingLabels.some(l => !l)) throw new Error('All images must have labels')
        
              // Create a training data bundle with images as base64 and labels
              const trainingData: { images: string[], labels: string[] } = { images: [], labels: trainingLabels }
              for (const img of trainingImages) {
                const base64 = await new Promise<string>((resolve) => {
                  const reader = new FileReader()
                  reader.onload = () => resolve(reader.result as string)
                  reader.readAsDataURL(img)
                })
                trainingData.images.push(base64)
              }
        
              // Upload as JSON blob
              const blob = new Blob([JSON.stringify(trainingData)], { type: 'application/json' })
              const formData = new FormData()
              formData.append('file', blob, 'training_data.json')
              const uploadRes = await fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData
              })
              if (!uploadRes.ok) throw new Error('Training data upload failed')
              const result = await uploadRes.json()
              dataId = result.data_id
            }
      
            if (!dataId && selectedKernel?.input_type !== 'matrix') {
              throw new Error('No data uploaded')
            }
      
            // Build params based on kernel type
            let params: Record<string, unknown> = {}
            if (kernelType.startsWith('image_')) {
              params = { grid_size: parseInt(gridSize) }
            } else if (kernelType === 'file_hash') {
              params = { chunk_size: 65536 }
            } else if (kernelType === 'text_wordcount') {
              params = { lines_per_chunk: 100 }
            } else if (kernelType === 'matrix_multiply') {
              params = { matrix_size: parseInt(matrixSize), block_size: parseInt(blockSize) }
            } else if (kernelType === 'model_finetune') {
              params = { 
                num_rounds: parseInt(numRounds), 
                epochs_per_round: parseInt(epochsPerRound),
                num_images: trainingImages.length,
                labels: [...new Set(trainingLabels)] // Unique labels
              }
            }
      
      // Create job using generic endpoint
      const jobRes = await fetch(
        `${API_URL}/jobs/create?data_id=${dataId}&kernel_type=${kernelType}&customer_id=demo-customer&params=${encodeURIComponent(JSON.stringify(params))}`,
        { method: 'POST' }
      )
      
      if (!jobRes.ok) throw new Error('Job creation failed')
      
            // Reset form
            setSelectedFile(null)
            setPreviewUrl(null)
            setTextContent('')
            setTrainingImages([])
            setTrainingLabels([])
            if (fileInputRef.current) fileInputRef.current.value = ''
      
      fetchJobs()
    } catch (err) {
      console.error('Failed to submit job:', err)
      alert(`Failed to submit job: ${err}`)
    }
    setIsUploading(false)
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge className="bg-green-600">Completed</Badge>
      case 'processing':
        return <Badge className="bg-blue-600">Processing</Badge>
      case 'queued':
        return <Badge className="bg-yellow-600">Queued</Badge>
      case 'decomposing':
        return <Badge className="bg-purple-600">Decomposing</Badge>
      case 'validating':
        return <Badge className="bg-cyan-600">Validating</Badge>
      case 'aggregating':
        return <Badge className="bg-indigo-600">Aggregating</Badge>
      case 'failed':
        return <Badge className="bg-red-600">Failed</Badge>
      default:
        return <Badge variant="outline">{status}</Badge>
    }
  }

    const getKernelIcon = (type: KernelType) => {
      if (type === 'model_finetune') return <Loader2 className="h-4 w-4" />
      if (type.startsWith('image_')) return <Image className="h-4 w-4" />
      if (type === 'file_hash') return <Hash className="h-4 w-4" />
      if (type === 'text_wordcount') return <FileText className="h-4 w-4" />
      if (type === 'matrix_multiply') return <Grid3X3 className="h-4 w-4" />
      return <File className="h-4 w-4" />
    }

    const getTaskCount = () => {
      if (kernelType === 'model_finetune') {
        return parseInt(numRounds)  // One fragment per training round
      } else if (kernelType === 'image_classify') {
        return 1  // One classification per image
      } else if (kernelType.startsWith('image_')) {
        return parseInt(gridSize) * parseInt(gridSize)
      } else if (kernelType === 'matrix_multiply') {
        const blocks = Math.ceil(parseInt(matrixSize) / parseInt(blockSize))
        return blocks * blocks * blocks
      }
      return '~'
    }

    const canSubmit = () => {
      if (kernelType === 'model_finetune') {
        return trainingImages.length >= 2 && trainingLabels.every(l => l.length > 0)
      }
      if (kernelType.startsWith('image_') || kernelType === 'file_hash') {
        return selectedFile !== null
      }
      if (kernelType === 'text_wordcount') {
        return selectedFile !== null || textContent.length > 0
      }
      if (kernelType === 'matrix_multiply') {
        return true
      }
      return false
    }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            Oxygen<span className="text-purple-400">TM</span>
          </h1>
          <p className="text-slate-300 text-lg">Customer Portal</p>
          <p className="text-sm text-slate-400 mt-1">
            Submit compute tasks for distributed processing across the network
          </p>
          <a 
            href="/worker.html" 
            className="inline-flex items-center gap-1 text-purple-400 hover:text-purple-300 text-sm mt-2"
          >
            Switch to Worker Portal <ExternalLink className="h-3 w-3" />
          </a>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upload Section */}
          <Card className="bg-white/95">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Submit Compute Job
              </CardTitle>
              <CardDescription>
                Choose a compute kernel and upload your data
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Kernel Type Selector */}
              <div className="space-y-2">
                <Label>Compute Kernel</Label>
                <Select value={kernelType} onValueChange={(v) => {
                  setKernelType(v as KernelType)
                  setSelectedFile(null)
                  setPreviewUrl(null)
                  setTextContent('')
                  if (fileInputRef.current) fileInputRef.current.value = ''
                }}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {KERNELS.map((kernel) => (
                      <SelectItem key={kernel.type} value={kernel.type}>
                        <div className="flex items-center gap-2">
                          {getKernelIcon(kernel.type)}
                          <span>{kernel.name}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {selectedKernel && (
                  <p className="text-xs text-gray-500">{selectedKernel.description}</p>
                )}
              </div>

              {/* Dynamic Upload Section based on kernel type */}
              {selectedKernel?.input_type === 'image' && (
                <div 
                  className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-purple-500 transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  {previewUrl ? (
                    <div className="space-y-2">
                      <img src={previewUrl} alt="Preview" className="max-h-48 mx-auto rounded" />
                      <p className="text-sm text-gray-600">{selectedFile?.name}</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <Image className="h-12 w-12 mx-auto text-gray-400" />
                      <p className="text-gray-600">Click to upload an image</p>
                      <p className="text-xs text-gray-400">PNG, JPG up to 5MB</p>
                    </div>
                  )}
                </div>
              )}

              {selectedKernel?.input_type === 'any' && (
                <div 
                  className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-purple-500 transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  {selectedFile ? (
                    <div className="space-y-2">
                      <File className="h-12 w-12 mx-auto text-purple-500" />
                      <p className="text-sm text-gray-600">{selectedFile.name}</p>
                      <p className="text-xs text-gray-400">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <File className="h-12 w-12 mx-auto text-gray-400" />
                      <p className="text-gray-600">Click to upload any file</p>
                      <p className="text-xs text-gray-400">File will be split into chunks for distributed hashing</p>
                    </div>
                  )}
                </div>
              )}

              {selectedKernel?.input_type === 'text' && (
                <div className="space-y-3">
                  <div 
                    className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-purple-500 transition-colors"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".txt,.csv,.md,.json"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                    {selectedFile ? (
                      <div className="flex items-center justify-center gap-2">
                        <FileText className="h-5 w-5 text-purple-500" />
                        <span className="text-sm text-gray-600">{selectedFile.name}</span>
                      </div>
                    ) : (
                      <p className="text-sm text-gray-500">Click to upload a text file, or type below</p>
                    )}
                  </div>
                  <Textarea
                    placeholder="Or paste/type text content here..."
                    value={textContent}
                    onChange={(e) => setTextContent(e.target.value)}
                    className="min-h-32"
                  />
                </div>
              )}

              {selectedKernel?.input_type === 'matrix' && (
                <div className="bg-gray-50 rounded-lg p-4 text-center">
                  <Grid3X3 className="h-12 w-12 mx-auto text-purple-500 mb-2" />
                  <p className="text-sm text-gray-600">Random matrices will be generated</p>
                  <p className="text-xs text-gray-400 mt-1">
                    Two {matrixSize}x{matrixSize} matrices will be multiplied using {blockSize}x{blockSize} blocks
                  </p>
                </div>
              )}

              {/* Dynamic Parameters based on kernel type */}
              {kernelType.startsWith('image_') && (
                <div className="space-y-2">
                  <Label>Grid Size (tiles)</Label>
                  <Select value={gridSize} onValueChange={setGridSize}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="2">2x2 (4 tiles)</SelectItem>
                      <SelectItem value="3">3x3 (9 tiles)</SelectItem>
                      <SelectItem value="4">4x4 (16 tiles)</SelectItem>
                      <SelectItem value="5">5x5 (25 tiles)</SelectItem>
                      <SelectItem value="8">8x8 (64 tiles)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}

                            {kernelType === 'matrix_multiply' && (
                              <div className="grid grid-cols-2 gap-3">
                                <div className="space-y-2">
                                  <Label>Matrix Size</Label>
                                  <Select value={matrixSize} onValueChange={setMatrixSize}>
                                    <SelectTrigger>
                                      <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="32">32x32</SelectItem>
                                      <SelectItem value="64">64x64</SelectItem>
                                      <SelectItem value="128">128x128</SelectItem>
                                      <SelectItem value="256">256x256</SelectItem>
                                    </SelectContent>
                                  </Select>
                                </div>
                                <div className="space-y-2">
                                  <Label>Block Size</Label>
                                  <Select value={blockSize} onValueChange={setBlockSize}>
                                    <SelectTrigger>
                                      <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="8">8x8</SelectItem>
                                      <SelectItem value="16">16x16</SelectItem>
                                      <SelectItem value="32">32x32</SelectItem>
                                    </SelectContent>
                                  </Select>
                                </div>
                              </div>
                            )}

                            {/* Training Data Upload for Fine-tuning */}
                            {selectedKernel?.input_type === 'training' && (
                              <div className="space-y-4">
                                <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                                  <h4 className="font-medium text-purple-800 mb-2">Distributed Fine-Tuning</h4>
                                  <p className="text-sm text-purple-600">
                                    Upload images with labels to fine-tune MobileNet. Workers will train on your data 
                                    and submit weight updates. Majority voting ensures training integrity.
                                  </p>
                                </div>

                                <div className="space-y-2">
                                  <Label>Training Images ({trainingImages.length} uploaded)</Label>
                                  <div 
                                    className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-purple-500 transition-colors"
                                    onClick={() => {
                                      const input = document.createElement('input')
                                      input.type = 'file'
                                      input.accept = 'image/*'
                                      input.multiple = true
                                      input.onchange = (e) => handleTrainingImageAdd(e as unknown as React.ChangeEvent<HTMLInputElement>)
                                      input.click()
                                    }}
                                  >
                                    <Upload className="h-8 w-8 mx-auto text-gray-400 mb-2" />
                                    <p className="text-sm text-gray-600">Click to add training images</p>
                                    <p className="text-xs text-gray-400">Add at least 2 images with labels</p>
                                  </div>
                                </div>

                                {trainingImages.length > 0 && (
                                  <div className="space-y-2 max-h-48 overflow-y-auto">
                                    {trainingImages.map((img, idx) => (
                                      <div key={idx} className="flex items-center gap-3 bg-gray-50 p-2 rounded">
                                        <img 
                                          src={URL.createObjectURL(img)} 
                                          alt={`Training ${idx}`} 
                                          className="w-12 h-12 object-cover rounded"
                                        />
                                        <input
                                          type="text"
                                          placeholder="Label (e.g., cat, dog)"
                                          value={trainingLabels[idx] || ''}
                                          onChange={(e) => handleLabelChange(idx, e.target.value)}
                                          className="flex-1 px-2 py-1 text-sm border rounded"
                                        />
                                        <button
                                          onClick={() => removeTrainingImage(idx)}
                                          className="text-red-500 hover:text-red-700 text-sm"
                                        >
                                          Remove
                                        </button>
                                      </div>
                                    ))}
                                  </div>
                                )}

                                <div className="grid grid-cols-2 gap-3">
                                  <div className="space-y-2">
                                    <Label>Training Rounds</Label>
                                    <Select value={numRounds} onValueChange={setNumRounds}>
                                      <SelectTrigger>
                                        <SelectValue />
                                      </SelectTrigger>
                                      <SelectContent>
                                        <SelectItem value="3">3 rounds</SelectItem>
                                        <SelectItem value="5">5 rounds</SelectItem>
                                        <SelectItem value="10">10 rounds</SelectItem>
                                        <SelectItem value="20">20 rounds</SelectItem>
                                      </SelectContent>
                                    </Select>
                                  </div>
                                  <div className="space-y-2">
                                    <Label>Epochs per Round</Label>
                                    <Select value={epochsPerRound} onValueChange={setEpochsPerRound}>
                                      <SelectTrigger>
                                        <SelectValue />
                                      </SelectTrigger>
                                      <SelectContent>
                                        <SelectItem value="1">1 epoch</SelectItem>
                                        <SelectItem value="2">2 epochs</SelectItem>
                                        <SelectItem value="5">5 epochs</SelectItem>
                                      </SelectContent>
                                    </Select>
                                  </div>
                                </div>
                              </div>
                            )}

                            <Button
                onClick={handleSubmit} 
                disabled={!canSubmit() || isUploading}
                className="w-full bg-purple-600 hover:bg-purple-700"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Submit Job'
                )}
              </Button>

              <p className="text-xs text-gray-500 text-center">
                Estimated tasks: {getTaskCount()} x $0.0001/task
              </p>
            </CardContent>
          </Card>

          {/* Jobs List */}
          <Card className="bg-white/95">
            <CardHeader>
              <CardTitle>Your Jobs</CardTitle>
              <CardDescription>
                Track processing progress and view results
              </CardDescription>
            </CardHeader>
            <CardContent>
              {jobs.length === 0 ? (
                <p className="text-center text-gray-500 py-8">No jobs submitted yet</p>
              ) : (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {jobs.map((job) => (
                    <div 
                      key={job.id}
                      className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedJob?.id === job.id ? 'border-purple-500 bg-purple-50' : 'hover:bg-gray-50'
                      }`}
                      onClick={() => fetchJobDetails(job.id)}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center gap-2">
                          {getKernelIcon(job.kernel_type)}
                          <div>
                            <p className="font-medium text-sm">
                              {KERNELS.find(k => k.type === job.kernel_type)?.name || job.kernel_type}
                            </p>
                            <p className="text-xs text-gray-500">
                              {new Date(job.created_at * 1000).toLocaleString()}
                            </p>
                          </div>
                        </div>
                        {getStatusBadge(job.status)}
                      </div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs text-gray-600">
                          <span>Progress</span>
                          <span>{job.verified_fragments} / {job.total_fragments} fragments verified</span>
                        </div>
                        <Progress 
                          value={job.total_fragments > 0 ? (job.verified_fragments / job.total_fragments) * 100 : 0} 
                          className="h-2"
                        />
                        {job.escrow_amount > 0 && (
                          <div className="text-xs text-gray-500">
                            Escrow: ${job.escrow_amount.toFixed(4)}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Job Details / Results */}
        {selectedJob && (
          <Card className="mt-6 bg-white/95">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {getKernelIcon(selectedJob.kernel_type)}
                Job Results
              </CardTitle>
              <CardDescription>
                {KERNELS.find(k => k.type === selectedJob.kernel_type)?.name || selectedJob.kernel_type}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Image processing results */}
              {selectedJob.kernel_type.startsWith('image_') && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-medium mb-2">Original Image</h3>
                    <img 
                      src={`${API_URL}/images/${selectedJob.data_id}`}
                      alt="Original"
                      className="w-full rounded border"
                    />
                  </div>
                  <div>
                    <h3 className="font-medium mb-2">
                      Processed Tiles 
                      <span className="text-sm font-normal text-gray-500 ml-2">
                        ({selectedJob.verified_fragments}/{selectedJob.total_fragments} verified)
                      </span>
                    </h3>
                    <div 
                      className="grid gap-1 border rounded p-1 bg-gray-100"
                      style={{ 
                        gridTemplateColumns: `repeat(${(selectedJob.params?.grid_size as number) || 4}, 1fr)` 
                      }}
                    >
                      {selectedJob.tasks?.map((task) => {
                        const row = task.shard_params?.row as number ?? 0
                        const col = task.shard_params?.col as number ?? 0
                        return (
                          <div 
                            key={task.id}
                            className="aspect-square relative bg-white rounded overflow-hidden"
                          >
                            {task.status === 'verified' || task.status === 'completed' ? (
                              <>
                                <img 
                                  src={`${API_URL}/jobs/${selectedJob.id}/result-tile/${row}/${col}`}
                                  alt={`Tile ${row},${col}`}
                                  className="w-full h-full object-cover"
                                />
                                {task.verified && (
                                  <div className="absolute top-0.5 right-0.5">
                                    <CheckCircle className="h-3 w-3 text-green-500" />
                                  </div>
                                )}
                              </>
                            ) : task.status === 'assigned' ? (
                              <div className="w-full h-full flex items-center justify-center bg-blue-50">
                                <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                              </div>
                            ) : (
                              <div className="w-full h-full flex items-center justify-center bg-gray-50">
                                <Clock className="h-4 w-4 text-gray-400" />
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>
              )}

              {/* File hash results */}
              {selectedJob.kernel_type === 'file_hash' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-gray-500">Total Chunks</p>
                      <p className="font-mono text-lg">{selectedJob.total_fragments}</p>
                    </div>
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-gray-500">Verified</p>
                      <p className="font-mono text-lg">{selectedJob.verified_fragments}</p>
                    </div>
                  </div>
                  {selectedJob.result?.merkle_root ? (
                    <div className="bg-green-50 p-3 rounded border border-green-200">
                      <p className="text-sm text-green-700 font-medium">Merkle Root Hash</p>
                      <p className="font-mono text-xs break-all mt-1">{String(selectedJob.result.merkle_root)}</p>
                    </div>
                  ) : null}
                  <div className="space-y-2">
                    <p className="font-medium text-sm">Chunk Hashes:</p>
                    <div className="max-h-48 overflow-y-auto space-y-1">
                      {selectedJob.tasks?.filter(t => t.result_hash).map((task) => (
                        <div key={task.id} className="flex justify-between text-xs bg-gray-50 p-2 rounded font-mono">
                          <span className="text-gray-500">{task.shard_id}</span>
                          <span className="truncate ml-2">{task.result_hash?.slice(0, 16)}...</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Text word count results */}
              {selectedJob.kernel_type === 'text_wordcount' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-gray-500">Total Chunks</p>
                      <p className="font-mono text-lg">{selectedJob.total_fragments}</p>
                    </div>
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-gray-500">Total Words</p>
                      <p className="font-mono text-lg">{(selectedJob.result?.total_words as number) || 0}</p>
                    </div>
                  </div>
                  {selectedJob.result?.word_counts && Object.keys(selectedJob.result.word_counts as Record<string, number>).length > 0 ? (
                    <div className="space-y-2">
                      <p className="font-medium text-sm">Top Words:</p>
                      <div className="grid grid-cols-2 gap-2">
                        {Object.entries(selectedJob.result.word_counts as Record<string, number>)
                          .sort(([, a], [, b]) => b - a)
                          .slice(0, 20)
                          .map(([word, count]) => (
                            <div key={word} className="flex justify-between text-sm bg-gray-50 p-2 rounded">
                              <span className="font-medium">{word}</span>
                              <span className="text-gray-500">{count}</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              )}

              {/* Matrix multiply results */}
              {selectedJob.kernel_type === 'matrix_multiply' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-gray-500">Matrix Size</p>
                      <p className="font-mono text-lg">{selectedJob.params?.matrix_size as number}x{selectedJob.params?.matrix_size as number}</p>
                    </div>
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-gray-500">Block Size</p>
                      <p className="font-mono text-lg">{selectedJob.params?.block_size as number}x{selectedJob.params?.block_size as number}</p>
                    </div>
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-gray-500">Total Blocks</p>
                      <p className="font-mono text-lg">{selectedJob.total_fragments}</p>
                    </div>
                  </div>
                  <Progress value={selectedJob.total_fragments > 0 ? (selectedJob.verified_fragments / selectedJob.total_fragments) * 100 : 0} className="h-3" />
                  <p className="text-sm text-gray-500 text-center">
                    {selectedJob.verified_fragments} / {selectedJob.total_fragments} blocks verified
                  </p>
                          {selectedJob.status === 'completed' && (
                            <div className="bg-green-50 p-3 rounded border border-green-200 text-center">
                              <CheckCircle className="h-8 w-8 text-green-500 mx-auto mb-2" />
                              <p className="text-green-700 font-medium">Matrix multiplication complete!</p>
                              <p className="text-xs text-green-600 mt-1">Result checksum: {(selectedJob.result?.checksum as string) || 'N/A'}</p>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Model fine-tuning results */}
                      {selectedJob.kernel_type === 'model_finetune' && (
                        <div className="space-y-4">
                          <div className="grid grid-cols-3 gap-4 text-sm">
                            <div className="bg-gray-50 p-3 rounded">
                              <p className="text-gray-500">Training Rounds</p>
                              <p className="font-mono text-lg">{selectedJob.params?.num_rounds as number || 5}</p>
                            </div>
                            <div className="bg-gray-50 p-3 rounded">
                              <p className="text-gray-500">Epochs/Round</p>
                              <p className="font-mono text-lg">{selectedJob.params?.epochs_per_round as number || 1}</p>
                            </div>
                            <div className="bg-gray-50 p-3 rounded">
                              <p className="text-gray-500">Training Images</p>
                              <p className="font-mono text-lg">{selectedJob.params?.num_images as number || 0}</p>
                            </div>
                          </div>
                  
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span>Training Progress</span>
                              <span>{selectedJob.verified_fragments} / {selectedJob.total_fragments} rounds complete</span>
                            </div>
                            <Progress value={selectedJob.total_fragments > 0 ? (selectedJob.verified_fragments / selectedJob.total_fragments) * 100 : 0} className="h-3" />
                          </div>

                                                    {selectedJob.result?.training_rounds ? (
                                                      <div className="space-y-2">
                                                        <p className="font-medium text-sm">Training Rounds:</p>
                                                        <div className="max-h-48 overflow-y-auto space-y-1">
                                                          {(selectedJob.result.training_rounds as Array<{round: number, loss_before: number, loss_after: number, accuracy: number}>).map((round) => (
                                                            <div key={round.round} className="flex justify-between text-xs bg-gray-50 p-2 rounded">
                                                              <span className="font-medium">Round {round.round + 1}</span>
                                                              <span className="text-gray-500">
                                                                Loss: {round.loss_before.toFixed(4)} → {round.loss_after.toFixed(4)} | 
                                                                Accuracy: {(round.accuracy * 100).toFixed(1)}%
                                                              </span>
                                                            </div>
                                                          ))}
                                                        </div>
                                                      </div>
                                                    ) : null}

                          {selectedJob.status === 'completed' && (
                            <div className="bg-green-50 p-4 rounded border border-green-200 text-center">
                              <CheckCircle className="h-8 w-8 text-green-500 mx-auto mb-2" />
                              <p className="text-green-700 font-medium">Model Fine-Tuning Complete!</p>
                              <div className="mt-2 text-sm text-green-600">
                                <p>Final Accuracy: {((selectedJob.result?.final_accuracy as number || 0) * 100).toFixed(1)}%</p>
                                <p>Total Loss Improvement: {(selectedJob.result?.total_loss_improvement as number || 0).toFixed(4)}</p>
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Image classification results */}
                      {selectedJob.kernel_type === 'image_classify' && (
                        <div className="space-y-4">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                              <h3 className="font-medium mb-2">Original Image</h3>
                              <img 
                                src={`${API_URL}/images/${selectedJob.data_id}`}
                                alt="Original"
                                className="w-full rounded border"
                              />
                            </div>
                            <div>
                              <h3 className="font-medium mb-2">Classification Results</h3>
                                                            {selectedJob.result?.classifications ? (
                                                              <div className="space-y-2">
                                                                {(selectedJob.result.classifications as Array<{top_class: string, confidence: number, predictions: Array<{className: string, probability: number}>}>).map((cls, idx) => (
                                                                  <div key={idx} className="bg-gray-50 p-3 rounded">
                                                                    <div className="flex justify-between items-center mb-2">
                                                                      <span className="font-medium text-lg">{cls.top_class}</span>
                                                                      <Badge className="bg-green-600">{(cls.confidence * 100).toFixed(1)}%</Badge>
                                                                    </div>
                                                                    {cls.predictions?.slice(0, 5).map((pred, pidx) => (
                                                                      <div key={pidx} className="flex justify-between text-xs text-gray-600">
                                                                        <span>{pred.className}</span>
                                                                        <span>{(pred.probability * 100).toFixed(1)}%</span>
                                                                      </div>
                                                                    ))}
                                                                  </div>
                                                                ))}
                                                              </div>
                                                            ) : null}
                            </div>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}
      </div>
    </div>
  )
}

export default CustomerPortal
