import React, { useState, useRef } from 'react'
import { useTensorStore } from '@/stores/tensorStore'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { PlusCircle, Upload, FileUp, X, AlertCircle, ChevronDown } from 'lucide-react'

const SHAPE_PRESETS = [
  { label: '2×2',     value: '2,2',   hint: 'cat / stack / permute' },
  { label: '3×3',     value: '3,3',   hint: 'reshape / narrow / chunk' },
  { label: '4×3',     value: '4,3',   hint: 'flatten / tile / repeat' },
  { label: '2×1×2',   value: '2,1,2', hint: 'squeeze / unsqueeze' },
  { label: '1×3×3',   value: '1,3,3', hint: '3-D permute / flatten' },
]

const ACCEPT = '.pt,.pth,.npy,.npz,.csv'

function FileDropZone({ onFile }) {
  const inputRef  = useRef(null)
  const [drag, setDrag] = useState(false)

  function handleDrop(e) {
    e.preventDefault(); setDrag(false)
    const f = e.dataTransfer.files?.[0]
    if (f) onFile(f)
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={[
        'flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed',
        'cursor-pointer px-4 py-6 text-center transition-colors select-none',
        drag
          ? 'border-primary bg-primary/5'
          : 'border-zinc-200 bg-zinc-50 hover:border-primary/50 hover:bg-zinc-100',
      ].join(' ')}
    >
      <FileUp className="h-6 w-6 text-zinc-400" />
      <p className="text-sm text-zinc-500">
        Drop file here or <span className="text-primary font-medium">browse</span>
      </p>
      <p className="text-[11px] text-zinc-400">.pt · .pth · .npy · .npz · .csv · max 50 MB</p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f) }}
      />
    </div>
  )
}

export default function TensorCreator() {
  const createTensor         = useTensorStore((s) => s.createTensor)
  const importTensor         = useTensorStore((s) => s.importTensor)
  const importTensorWithKey  = useTensorStore((s) => s.importTensorWithKey)
  const importKeys           = useTensorStore((s) => s.importKeys)

  const [tab, setTab]             = useState('create')   // 'create' | 'import'
  const [op,  setOp]              = useState('ones')
  const [shapeInput, setShapeInput] = useState('3,3')

  // Import state
  const [file,        setFile]        = useState(null)
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState(null)
  const [selectedKey, setSelectedKey] = useState('')

  // ── Create handler ─────────────────────────────────────────────────────────
  async function handleCreate() {
    const shape = shapeInput.split(',').map(Number)
    await createTensor(op, shape)
  }

  // ── File selected ───────────────────────────────────────────────────────
  function handleFileSelect(f) {
    setFile(f); setError(null); setSelectedKey('')
  }

  function handleClearFile() {
    setFile(null); setError(null); setSelectedKey('')
  }

  // ── Load button ──────────────────────────────────────────────────────────
  async function handleLoad() {
    if (!file) return
    setLoading(true); setError(null)
    try {
      const result = await importTensor(file)
      if (!result.needsKey) {
        // Success — reset local file state
        setFile(null); setSelectedKey('')
      }
      // If needsKey=true, importKeys in store is now populated; UI shows key picker
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // ── Key confirm button ────────────────────────────────────────────────────
  async function handleKeyConfirm() {
    if (!selectedKey) return
    setLoading(true); setError(null)
    try {
      await importTensorWithKey(selectedKey)
      setFile(null); setSelectedKey('')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const needsKeyPicker = importKeys.length > 0

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Tensor Source</CardTitle>
        {/* Tab switcher */}
        <div className="flex rounded-md border border-zinc-200 overflow-hidden mt-1">
          {['create', 'import'].map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => { setTab(t); setError(null) }}
              className={[
                'flex-1 py-1.5 text-xs font-medium capitalize transition-colors',
                tab === t
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-white text-muted-foreground hover:bg-zinc-50',
              ].join(' ')}
            >
              {t === 'create' ? 'Create' : 'Import'}
            </button>
          ))}
        </div>
      </CardHeader>

      <CardContent className="space-y-3">

        {/* ── CREATE TAB ─────────────────────────────────────────────────── */}
        {tab === 'create' && (
          <>
            <div className="space-y-1">
              <Label>Initializer</Label>
              <Select value={op} onValueChange={setOp}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="ones">Ones</SelectItem>
                  <SelectItem value="zeros">Zeros</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1">
              <Label>Shape (comma-separated)</Label>
              <Input value={shapeInput} onChange={(e) => setShapeInput(e.target.value)}
                placeholder="e.g. 3,3" className="font-mono" />
            </div>

            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">Quick presets</Label>
              <div className="flex flex-wrap gap-1.5">
                {SHAPE_PRESETS.map((p) => (
                  <button key={p.value} type="button" title={p.hint}
                    onClick={() => setShapeInput(p.value)}
                    className={[
                      'rounded border px-2 py-0.5 text-xs font-mono transition-colors',
                      shapeInput === p.value
                        ? 'border-primary bg-primary text-primary-foreground'
                        : 'border-border bg-muted text-muted-foreground hover:border-primary hover:text-foreground',
                    ].join(' ')}>
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            <Button className="w-full" onClick={handleCreate}>
              <PlusCircle className="mr-2 h-4 w-4" />Create
            </Button>
          </>
        )}

        {/* ── IMPORT TAB ─────────────────────────────────────────────────── */}
        {tab === 'import' && (
          <>
            {/* Error banner */}
            {error && (
              <div className="flex items-start gap-2 rounded-md border border-red-200 bg-red-50 p-2 text-xs text-red-700">
                <AlertCircle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Drop zone or selected file */}
            {!file ? (
              <FileDropZone onFile={handleFileSelect} />
            ) : (
              <div className="flex items-center justify-between gap-2 rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-zinc-700">{file.name}</p>
                  <p className="text-[11px] text-zinc-400">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
                <button type="button" onClick={handleClearFile}
                  className="shrink-0 rounded p-1 text-zinc-400 hover:bg-zinc-200 hover:text-zinc-600">
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            )}

            {/* Key picker (shown after dict file is uploaded) */}
            {needsKeyPicker && (
              <div className="space-y-2 rounded-md border border-amber-200 bg-amber-50 p-3">
                <p className="text-xs font-medium text-amber-800">
                  This file contains multiple tensors. Select which one to load:
                </p>
                <Select value={selectedKey} onValueChange={setSelectedKey}>
                  <SelectTrigger className="bg-white text-xs">
                    <SelectValue placeholder="Choose a key…" />
                  </SelectTrigger>
                  <SelectContent className="max-h-48 overflow-y-auto">
                    {importKeys.map((k) => (
                      <SelectItem key={k} value={k} className="font-mono text-xs">{k}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button className="w-full" size="sm"
                  disabled={!selectedKey || loading}
                  onClick={handleKeyConfirm}>
                  {loading ? 'Loading…' : 'Load selected tensor'}
                </Button>
              </div>
            )}

            {/* Load button (shown before key picker) */}
            {!needsKeyPicker && (
              <Button className="w-full" disabled={!file || loading} onClick={handleLoad}>
                <Upload className="mr-2 h-4 w-4" />
                {loading ? 'Loading…' : 'Load Tensor'}
              </Button>
            )}
          </>
        )}

      </CardContent>
    </Card>
  )
}
