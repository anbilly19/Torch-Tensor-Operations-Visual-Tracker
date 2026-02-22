import React, { useState, useEffect } from 'react'
import { useTensorStore } from '@/stores/tensorStore'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Play, BookOpen } from 'lucide-react'

// ── Docs map ───────────────────────────────────────────────────────────────────
const DOCS = {
  add:                'https://pytorch.org/docs/stable/generated/torch.add.html',
  sub:                'https://pytorch.org/docs/stable/generated/torch.sub.html',
  mul:                'https://pytorch.org/docs/stable/generated/torch.mul.html',
  div:                'https://pytorch.org/docs/stable/generated/torch.div.html',
  matmul:             'https://pytorch.org/docs/stable/generated/torch.matmul.html',
  abs:                'https://pytorch.org/docs/stable/generated/torch.abs.html',
  neg:                'https://pytorch.org/docs/stable/generated/torch.neg.html',
  clamp:              'https://pytorch.org/docs/stable/generated/torch.clamp.html',
  sum:                'https://pytorch.org/docs/stable/generated/torch.sum.html',
  reshape:            'https://pytorch.org/docs/stable/generated/torch.reshape.html',
  transpose:          'https://pytorch.org/docs/stable/generated/torch.transpose.html',
  flatten:            'https://pytorch.org/docs/stable/generated/torch.flatten.html',
  squeeze:            'https://pytorch.org/docs/stable/generated/torch.squeeze.html',
  unsqueeze:          'https://pytorch.org/docs/stable/generated/torch.unsqueeze.html',
  permute:            'https://pytorch.org/docs/stable/generated/torch.permute.html',
  tile:               'https://pytorch.org/docs/stable/generated/torch.tile.html',
  repeat:             'https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html',
  narrow:             'https://pytorch.org/docs/stable/generated/torch.narrow.html',
  chunk:              'https://pytorch.org/docs/stable/generated/torch.chunk.html',
  cat:                'https://pytorch.org/docs/stable/generated/torch.cat.html',
  stack:              'https://pytorch.org/docs/stable/generated/torch.stack.html',
  linear:             'https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html',
  conv1d:             'https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html',
  conv2d:             'https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html',
  maxpool1d:          'https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool1d.html',
  maxpool2d:          'https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html',
  avgpool1d:          'https://pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool1d.html',
  avgpool2d:          'https://pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool2d.html',
  adaptive_avgpool2d: 'https://pytorch.org/docs/stable/generated/torch.nn.functional.adaptive_avg_pool2d.html',
  embedding:          'https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html',
  sdpa:               'https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html',
}

// ── Operation groups ────────────────────────────────────────────────────────────
const OPERATIONS = [
  {
    group: 'Elementwise / Scalar',
    ops: [
      { value: 'add',    label: 'Add',             binary: true  },
      { value: 'sub',    label: 'Subtract',        binary: true  },
      { value: 'mul',    label: 'Multiply',        binary: true  },
      { value: 'div',    label: 'Divide',          binary: true  },
      { value: 'matmul', label: 'Matrix Multiply', binary: true  },
      { value: 'abs',    label: 'Absolute Value',  binary: false },
      { value: 'neg',    label: 'Negate',          binary: false },
      { value: 'clamp',  label: 'Clamp',           binary: false },
    ],
  },
  {
    group: 'Reduction',
    ops: [
      { value: 'sum', label: 'Sum', binary: false },
    ],
  },
  {
    group: 'Shape & Indexing',
    ops: [
      { value: 'reshape',   label: 'Reshape',   binary: false },
      { value: 'transpose', label: 'Transpose', binary: false },
      { value: 'flatten',   label: 'Flatten',   binary: false },
      { value: 'squeeze',   label: 'Squeeze',   binary: false },
      { value: 'unsqueeze', label: 'Unsqueeze', binary: false },
      { value: 'permute',   label: 'Permute',   binary: false },
      { value: 'tile',      label: 'Tile',      binary: false },
      { value: 'repeat',    label: 'Repeat',    binary: false },
      { value: 'narrow',    label: 'Narrow',    binary: false },
      { value: 'chunk',     label: 'Chunk',     binary: false },
      { value: 'cat',       label: 'Cat',       binary: false },
      { value: 'stack',     label: 'Stack',     binary: false },
    ],
  },
  {
    group: 'Layers',
    ops: [
      { value: 'linear',             label: 'Linear',                        binary: false },
      { value: 'conv1d',             label: 'Conv1D',                        binary: false },
      { value: 'conv2d',             label: 'Conv2D',                        binary: false },
      { value: 'maxpool1d',          label: 'MaxPool1D',                     binary: false },
      { value: 'maxpool2d',          label: 'MaxPool2D',                     binary: false },
      { value: 'avgpool1d',          label: 'AvgPool1D',                     binary: false },
      { value: 'avgpool2d',          label: 'AvgPool2D',                     binary: false },
      { value: 'adaptive_avgpool2d', label: 'Adaptive AvgPool2D',            binary: false },
      { value: 'embedding',          label: 'Embedding',                     binary: false },
      { value: 'sdpa',               label: 'Scaled Dot-Product Attention',  binary: false },
    ],
  },
]

const ALL_OPS = OPERATIONS.flatMap((g) => g.ops)

// ── Shape hint helper ───────────────────────────────────────────────────────────
function getShape(tensor) {
  if (!Array.isArray(tensor)) return []
  const shape = []
  let t = tensor
  while (Array.isArray(t)) { shape.push(t.length); t = t[0] }
  return shape
}

// ── Component ───────────────────────────────────────────────────────────────────
export default function OperationPanel() {
  const applyOperation = useTensorStore((s) => s.applyOperation)
  const currentTensor  = useTensorStore((s) => s.currentTensor)
  const chunkTensors   = useTensorStore((s) => s.chunkTensors)

  const [selectedOp, setSelectedOp] = useState('add')

  // Elementwise
  const [tensorB,   setTensorB]   = useState('[[5,6],[7,8]]')
  const [clampMin,  setClampMin]  = useState('')
  const [clampMax,  setClampMax]  = useState('')

  // Reduction
  const [sumDim,   setSumDim]   = useState('')
  const [keepdim,  setKeepdim]  = useState(false)

  // Shape & indexing
  const [reshapeShape,  setReshapeShape]  = useState('1,9')
  const [dim0,          setDim0]          = useState('0')
  const [dim1,          setDim1]          = useState('1')
  const [flatStartDim,  setFlatStartDim]  = useState('0')
  const [flatEndDim,    setFlatEndDim]    = useState('-1')
  const [squeezeDim,    setSqueezeDim]    = useState('1')
  const [unsqueezeDim,  setUnsqueezeDim]  = useState('0')
  const [permuteDims,   setPermuteDims]   = useState('1,0')
  const [tileDims,      setTileDims]      = useState('2,2')
  const [repeatSizes,   setRepeatSizes]   = useState('2,2')
  const [narrowDim,     setNarrowDim]     = useState('0')
  const [narrowStart,   setNarrowStart]   = useState('1')
  const [narrowLength,  setNarrowLength]  = useState('1')
  const [chunkN,        setChunkN]        = useState('3')
  const [chunkDim,      setChunkDim]      = useState('0')
  const [catTensors,    setCatTensors]    = useState('[]')
  const [catDim,        setCatDim]        = useState('0')
  const [stackTensors,  setStackTensors]  = useState('[]')
  const [stackDim,      setStackDim]      = useState('0')

  // Layer ops
  const [linearOut,       setLinearOut]       = useState('8')
  const [convOutCh,       setConvOutCh]       = useState('4')
  const [convKernel,      setConvKernel]      = useState('3')
  const [convStride,      setConvStride]      = useState('1')
  const [convPadding,     setConvPadding]     = useState('0')
  const [poolKernel,      setPoolKernel]      = useState('2')
  const [poolStride,      setPoolStride]      = useState('')
  const [adaptiveOut,     setAdaptiveOut]     = useState('4,4')
  const [embedDim,        setEmbedDim]        = useState('8')
  const [embedVocab,      setEmbedVocab]      = useState('')
  const [sdpaKey,         setSdpaKey]         = useState('')
  const [sdpaValue,       setSdpaValue]       = useState('')

  // Sync cat/stack placeholders to current tensor shape
  useEffect(() => {
    if (!currentTensor) return
    function shapeLike(t) {
      if (!Array.isArray(t)) return 2
      return t.map(shapeLike)
    }
    const extra = JSON.stringify([shapeLike(currentTensor)])
    setCatTensors(extra)
    setStackTensors(extra)
  }, [currentTensor])

  // Seed SDPA K/V with same shape as current tensor
  useEffect(() => {
    if (!currentTensor) return
    const json = JSON.stringify(currentTensor)
    setSdpaKey(json)
    setSdpaValue(json)
  }, [currentTensor])

  const isBinary = ALL_OPS.find((o) => o.value === selectedOp)?.binary ?? false
  const docsUrl  = DOCS[selectedOp] ?? null
  const shape    = getShape(currentTensor)

  // ── Apply handler ───────────────────────────────────────────────────────────────
  async function handleApply() {
    let params = {}

    if (isBinary) {
      try { params.tensor_b = JSON.parse(tensorB) }
      catch { alert('Invalid JSON for second tensor'); return }
    }

    switch (selectedOp) {
      case 'clamp':
        if (clampMin !== '') params.min_val = Number(clampMin)
        if (clampMax !== '') params.max_val = Number(clampMax)
        break
      case 'sum':
        params.dim     = sumDim !== '' ? Number(sumDim) : null
        params.keepdim = keepdim
        break
      case 'reshape':
        params.shape = reshapeShape.split(',').map(Number)
        break
      case 'transpose':
        params.dim0 = Number(dim0); params.dim1 = Number(dim1)
        break
      case 'flatten':
        params.start_dim = Number(flatStartDim); params.end_dim = Number(flatEndDim)
        break
      case 'squeeze':
        params.dim = squeezeDim !== '' ? Number(squeezeDim) : undefined
        break
      case 'unsqueeze':
        params.dim = Number(unsqueezeDim)
        break
      case 'permute':
        params.dims = permuteDims.split(',').map(Number)
        break
      case 'tile':
        params.dims = tileDims.split(',').map(Number)
        break
      case 'repeat':
        params.sizes = repeatSizes.split(',').map(Number)
        break
      case 'narrow':
        params.dim = Number(narrowDim); params.start = Number(narrowStart); params.length = Number(narrowLength)
        break
      case 'chunk':
        params.chunks = Number(chunkN); params.dim = Number(chunkDim)
        break
      case 'cat': {
        let extras; try { extras = JSON.parse(catTensors) } catch { alert('Invalid JSON'); return }
        params.tensors = [currentTensor, ...extras]; params.dim = Number(catDim)
        break
      }
      case 'stack': {
        let extras; try { extras = JSON.parse(stackTensors) } catch { alert('Invalid JSON'); return }
        params.tensors = [currentTensor, ...extras]; params.dim = Number(stackDim)
        break
      }
      // ── Layer ops ─────────────────────────────────────────────────────────
      case 'linear':
        params.out_features = Number(linearOut)
        break
      case 'conv1d':
        params.out_channels  = Number(convOutCh)
        params.kernel_size   = Number(convKernel)
        params.stride        = Number(convStride)
        params.padding       = Number(convPadding)
        break
      case 'conv2d':
        params.out_channels  = Number(convOutCh)
        params.kernel_size   = Number(convKernel)
        params.stride        = Number(convStride)
        params.padding       = Number(convPadding)
        break
      case 'maxpool1d':
      case 'maxpool2d':
      case 'avgpool1d':
      case 'avgpool2d':
        params.kernel_size = Number(poolKernel)
        if (poolStride !== '') params.stride = Number(poolStride)
        break
      case 'adaptive_avgpool2d':
        params.output_size = adaptiveOut.split(',').map(Number)
        break
      case 'embedding':
        params.embed_dim  = Number(embedDim)
        if (embedVocab !== '') params.vocab_size = Number(embedVocab)
        break
      case 'sdpa': {
        let k, v
        try { k = JSON.parse(sdpaKey);   } catch { alert('Invalid JSON for Key tensor');   return }
        try { v = JSON.parse(sdpaValue); } catch { alert('Invalid JSON for Value tensor'); return }
        params.key   = k
        params.value = v
        break
      }
      default:
        break
    }

    await applyOperation(selectedOp, params)
  }

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Apply Operation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">

        {/* Selector + docs pill */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <Label>Operation</Label>
            {docsUrl && (
              <a href={docsUrl} target="_blank" rel="noopener noreferrer"
                className="inline-flex items-center gap-1 rounded-full border border-blue-200 bg-blue-50
                           px-2 py-0.5 text-[11px] font-medium text-blue-600
                           transition-colors hover:bg-blue-100 hover:text-blue-700">
                <BookOpen className="h-3 w-3" />PyTorch docs
              </a>
            )}
          </div>
          <Select value={selectedOp} onValueChange={setSelectedOp}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              {OPERATIONS.map((group) => (
                <SelectGroup key={group.group}>
                  <SelectLabel>{group.group}</SelectLabel>
                  {group.ops.map((op) => (
                    <SelectItem key={op.value} value={op.value}>{op.label}</SelectItem>
                  ))}
                </SelectGroup>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Shape hint */}
        {currentTensor && shape.length > 0 && (
          <p className="text-[11px] text-muted-foreground font-mono">
            Current shape: [{shape.join(', ')}]
          </p>
        )}

        <Separator />

        {/* ── Binary: second tensor ─────────────────────────────────────────── */}
        {isBinary && (
          <div className="space-y-1">
            <Label>Second tensor (JSON)</Label>
            <Textarea value={tensorB} onChange={(e) => setTensorB(e.target.value)}
              placeholder="e.g. [[1,2],[3,4]]" className="font-mono text-xs" />
          </div>
        )}

        {/* ── Clamp ──────────────────────────────────────────────────────────── */}
        {selectedOp === 'clamp' && (
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1"><Label>Min (optional)</Label>
              <Input type="number" value={clampMin} onChange={(e) => setClampMin(e.target.value)} placeholder="−∞" /></div>
            <div className="space-y-1"><Label>Max (optional)</Label>
              <Input type="number" value={clampMax} onChange={(e) => setClampMax(e.target.value)} placeholder="+∞" /></div>
          </div>
        )}

        {/* ── Sum ────────────────────────────────────────────────────────────── */}
        {selectedOp === 'sum' && (
          <div className="space-y-3">
            <div className="space-y-1"><Label>Dimension (empty = all)</Label>
              <Input type="number" value={sumDim} onChange={(e) => setSumDim(e.target.value)} placeholder="optional" /></div>
            <div className="flex items-center gap-2">
              <Checkbox id="keepdim" checked={keepdim} onCheckedChange={setKeepdim} />
              <Label htmlFor="keepdim">Keep dimensions</Label>
            </div>
          </div>
        )}

        {/* ── Reshape ─────────────────────────────────────────────────────────── */}
        {selectedOp === 'reshape' && (
          <div className="space-y-1"><Label>New shape (comma-separated)</Label>
            <Input value={reshapeShape} onChange={(e) => setReshapeShape(e.target.value)}
              placeholder="e.g. 1,9" className="font-mono" /></div>
        )}

        {/* ── Transpose ───────────────────────────────────────────────────────── */}
        {selectedOp === 'transpose' && (
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1"><Label>dim0</Label>
              <Input type="number" value={dim0} onChange={(e) => setDim0(e.target.value)} /></div>
            <div className="space-y-1"><Label>dim1</Label>
              <Input type="number" value={dim1} onChange={(e) => setDim1(e.target.value)} /></div>
          </div>
        )}

        {/* ── Flatten ─────────────────────────────────────────────────────────── */}
        {selectedOp === 'flatten' && (
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1"><Label>start_dim</Label>
              <Input type="number" value={flatStartDim} onChange={(e) => setFlatStartDim(e.target.value)} /></div>
            <div className="space-y-1"><Label>end_dim</Label>
              <Input type="number" value={flatEndDim} onChange={(e) => setFlatEndDim(e.target.value)} /></div>
          </div>
        )}

        {/* ── Squeeze ──────────────────────────────────────────────────────────── */}
        {selectedOp === 'squeeze' && (
          <div className="space-y-1"><Label>dim (empty = all size-1 dims)</Label>
            <Input type="number" value={squeezeDim} onChange={(e) => setSqueezeDim(e.target.value)} placeholder="optional" /></div>
        )}

        {/* ── Unsqueeze ───────────────────────────────────────────────────────── */}
        {selectedOp === 'unsqueeze' && (
          <div className="space-y-1"><Label>dim</Label>
            <Input type="number" value={unsqueezeDim} onChange={(e) => setUnsqueezeDim(e.target.value)} /></div>
        )}

        {/* ── Permute ──────────────────────────────────────────────────────────── */}
        {selectedOp === 'permute' && (
          <div className="space-y-1"><Label>dims (comma-separated, e.g. 1,0)</Label>
            <Input value={permuteDims} onChange={(e) => setPermuteDims(e.target.value)}
              placeholder="1,0" className="font-mono" /></div>
        )}

        {/* ── Tile ──────────────────────────────────────────────────────────────── */}
        {selectedOp === 'tile' && (
          <div className="space-y-1"><Label>dims — repeat count per axis</Label>
            <Input value={tileDims} onChange={(e) => setTileDims(e.target.value)}
              placeholder="2,2" className="font-mono" /></div>
        )}

        {/* ── Repeat ───────────────────────────────────────────────────────────── */}
        {selectedOp === 'repeat' && (
          <div className="space-y-1"><Label>sizes — copies per axis</Label>
            <Input value={repeatSizes} onChange={(e) => setRepeatSizes(e.target.value)}
              placeholder="2,2" className="font-mono" /></div>
        )}

        {/* ── Narrow ───────────────────────────────────────────────────────────── */}
        {selectedOp === 'narrow' && (
          <div className="grid grid-cols-3 gap-2">
            <div className="space-y-1"><Label>dim</Label>
              <Input type="number" value={narrowDim}    onChange={(e) => setNarrowDim(e.target.value)} /></div>
            <div className="space-y-1"><Label>start</Label>
              <Input type="number" value={narrowStart}  onChange={(e) => setNarrowStart(e.target.value)} /></div>
            <div className="space-y-1"><Label>length</Label>
              <Input type="number" value={narrowLength} onChange={(e) => setNarrowLength(e.target.value)} /></div>
          </div>
        )}

        {/* ── Chunk ────────────────────────────────────────────────────────────── */}
        {selectedOp === 'chunk' && (
          <>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1"><Label>chunks</Label>
                <Input type="number" value={chunkN}   onChange={(e) => setChunkN(e.target.value)}   min={1} /></div>
              <div className="space-y-1"><Label>dim</Label>
                <Input type="number" value={chunkDim} onChange={(e) => setChunkDim(e.target.value)} /></div>
            </div>
            {chunkTensors && (
              <div className="rounded-md border border-dashed border-amber-300 bg-amber-50 p-2 text-xs text-amber-800 space-y-1">
                <p className="font-medium">{chunkTensors.length} chunks — current = chunk[0]</p>
                {chunkTensors.map((c, i) => (
                  <p key={i} className="font-mono text-[11px]">chunk[{i}] [{c.shape.join(', ')}]</p>
                ))}
              </div>
            )}
          </>
        )}

        {/* ── Cat ──────────────────────────────────────────────────────────────── */}
        {selectedOp === 'cat' && (
          <div className="space-y-3">
            <div className="space-y-1">
              <Label>Extra tensors (JSON array — current tensor is first)</Label>
              <Textarea value={catTensors} onChange={(e) => setCatTensors(e.target.value)}
                placeholder="[[[1,2],[3,4]]]" className="font-mono text-xs" rows={4} /></div>
            <div className="space-y-1"><Label>dim</Label>
              <Input type="number" value={catDim} onChange={(e) => setCatDim(e.target.value)} /></div>
          </div>
        )}

        {/* ── Stack ───────────────────────────────────────────────────────────── */}
        {selectedOp === 'stack' && (
          <div className="space-y-3">
            <div className="space-y-1">
              <Label>Extra tensors (JSON array — all must match shape)</Label>
              <Textarea value={stackTensors} onChange={(e) => setStackTensors(e.target.value)}
                placeholder="[[[1,2],[3,4]]]" className="font-mono text-xs" rows={4} /></div>
            <div className="space-y-1"><Label>dim</Label>
              <Input type="number" value={stackDim} onChange={(e) => setStackDim(e.target.value)} /></div>
          </div>
        )}

        {/* ── Linear ───────────────────────────────────────────────────────────── */}
        {selectedOp === 'linear' && (
          <div className="space-y-1">
            <Label>out_features</Label>
            <Input type="number" value={linearOut} onChange={(e) => setLinearOut(e.target.value)}
              placeholder="e.g. 8" />
            <p className="text-[11px] text-muted-foreground">Input: [*, in] → Output: [*, {linearOut || '?'}]</p>
          </div>
        )}

        {/* ── Conv1D / Conv2D ────────────────────────────────────────────────── */}
        {(selectedOp === 'conv1d' || selectedOp === 'conv2d') && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1"><Label>out_channels</Label>
                <Input type="number" value={convOutCh}   onChange={(e) => setConvOutCh(e.target.value)} /></div>
              <div className="space-y-1"><Label>kernel_size</Label>
                <Input type="number" value={convKernel}  onChange={(e) => setConvKernel(e.target.value)} /></div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1"><Label>stride</Label>
                <Input type="number" value={convStride}  onChange={(e) => setConvStride(e.target.value)} /></div>
              <div className="space-y-1"><Label>padding</Label>
                <Input type="number" value={convPadding} onChange={(e) => setConvPadding(e.target.value)} /></div>
            </div>
            <p className="text-[11px] text-muted-foreground">
              {selectedOp === 'conv1d'
                ? 'Input: [N, C_in, L] → Output: [N, C_out, L_out]'
                : 'Input: [N, C_in, H, W] → Output: [N, C_out, H_out, W_out]'}
            </p>
          </div>
        )}

        {/* ── MaxPool / AvgPool (1D + 2D) ─────────────────────────────────────── */}
        {['maxpool1d','maxpool2d','avgpool1d','avgpool2d'].includes(selectedOp) && (
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1"><Label>kernel_size</Label>
              <Input type="number" value={poolKernel} onChange={(e) => setPoolKernel(e.target.value)} /></div>
            <div className="space-y-1"><Label>stride (empty = kernel_size)</Label>
              <Input type="number" value={poolStride} onChange={(e) => setPoolStride(e.target.value)}
                placeholder="auto" /></div>
          </div>
        )}

        {/* ── Adaptive AvgPool2D ──────────────────────────────────────────────── */}
        {selectedOp === 'adaptive_avgpool2d' && (
          <div className="space-y-1">
            <Label>output_size (H_out, W_out — comma-separated)</Label>
            <Input value={adaptiveOut} onChange={(e) => setAdaptiveOut(e.target.value)}
              placeholder="4,4" className="font-mono" />
            <p className="text-[11px] text-muted-foreground">Input: [N, C, H, W] → Output: [N, C, {adaptiveOut || '?'}]</p>
          </div>
        )}

        {/* ── Embedding ────────────────────────────────────────────────────────── */}
        {selectedOp === 'embedding' && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1"><Label>embed_dim</Label>
                <Input type="number" value={embedDim}   onChange={(e) => setEmbedDim(e.target.value)} /></div>
              <div className="space-y-1"><Label>vocab_size (optional)</Label>
                <Input type="number" value={embedVocab} onChange={(e) => setEmbedVocab(e.target.value)}
                  placeholder="auto" /></div>
            </div>
            <p className="text-[11px] text-muted-foreground">
              Input must be integer indices: [N, seq_len] → Output: [N, seq_len, {embedDim || '?'}]
            </p>
          </div>
        )}

        {/* ── SDPA ───────────────────────────────────────────────────────────────── */}
        {selectedOp === 'sdpa' && (
          <div className="space-y-3">
            <p className="text-[11px] text-muted-foreground">
              Current tensor is used as <strong>Query</strong>. Provide K and V below (same shape).
              All must be [N, H, T, D].
            </p>
            <div className="space-y-1">
              <Label>Key tensor (JSON)</Label>
              <Textarea value={sdpaKey}   onChange={(e) => setSdpaKey(e.target.value)}
                className="font-mono text-xs" rows={3} />
            </div>
            <div className="space-y-1">
              <Label>Value tensor (JSON)</Label>
              <Textarea value={sdpaValue} onChange={(e) => setSdpaValue(e.target.value)}
                className="font-mono text-xs" rows={3} />
            </div>
          </div>
        )}

        <Button className="w-full" onClick={handleApply}>
          <Play className="mr-2 h-4 w-4" />
          Apply
        </Button>

      </CardContent>
    </Card>
  )
}
