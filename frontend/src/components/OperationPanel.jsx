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
import { Play } from 'lucide-react'

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
]

const ALL_OPS = OPERATIONS.flatMap((g) => g.ops)

export default function OperationPanel() {
  const applyOperation  = useTensorStore((s) => s.applyOperation)
  const currentTensor   = useTensorStore((s) => s.currentTensor)
  const chunkTensors    = useTensorStore((s) => s.chunkTensors)

  const [selectedOp, setSelectedOp] = useState('add')

  // ── Elementwise / scalar ──────────────────────────────────────────────────
  const [tensorB, setTensorB]     = useState('[[5,6],[7,8]]')
  const [clampMin, setClampMin]   = useState('')
  const [clampMax, setClampMax]   = useState('')

  // ── Reduction ─────────────────────────────────────────────────────────────
  const [sumDim, setSumDim]       = useState('')
  const [keepdim, setKeepdim]     = useState(false)

  // ── Shape & indexing ──────────────────────────────────────────────────────
  const [reshapeShape, setReshapeShape]   = useState('1,9')
  const [dim0, setDim0]                   = useState('0')
  const [dim1, setDim1]                   = useState('1')
  const [flatStartDim, setFlatStartDim]   = useState('0')
  const [flatEndDim, setFlatEndDim]       = useState('-1')
  const [squeezeDim, setSqueezeDim]       = useState('1')
  const [unsqueezeDim, setUnsqueezeDim]   = useState('0')
  const [permuteDims, setPermuteDims]     = useState('1,0')
  const [tileDims, setTileDims]           = useState('2,2')
  const [repeatSizes, setRepeatSizes]     = useState('2,2')
  const [narrowDim, setNarrowDim]         = useState('0')
  const [narrowStart, setNarrowStart]     = useState('1')
  const [narrowLength, setNarrowLength]   = useState('1')
  const [chunkN, setChunkN]               = useState('3')
  const [chunkDim, setChunkDim]           = useState('0')
  // cat / stack: JSON array of tensors; currentTensor is pre-seeded as first
  const [catTensors, setCatTensors]       = useState('[]')
  const [catDim, setCatDim]               = useState('0')
  const [stackTensors, setStackTensors]   = useState('[]')
  const [stackDim, setStackDim]           = useState('0')

  // Keep cat/stack extra-tensor placeholders in sync with current tensor shape
  useEffect(() => {
    if (!currentTensor) return
    // Default extra tensor: same shape filled with 2s
    function shapeLike(t) {
      if (!Array.isArray(t)) return 2
      return t.map(shapeLike)
    }
    const extra = JSON.stringify(shapeLike(currentTensor).map ? shapeLike(currentTensor) : [[2]])
    setCatTensors(`[${extra}]`)
    setStackTensors(`[${extra}]`)
  }, [currentTensor])

  const isBinary = ALL_OPS.find((o) => o.value === selectedOp)?.binary ?? false

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
        params.dim0 = Number(dim0)
        params.dim1 = Number(dim1)
        break
      case 'flatten':
        params.start_dim = Number(flatStartDim)
        params.end_dim   = Number(flatEndDim)
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
        params.dim    = Number(narrowDim)
        params.start  = Number(narrowStart)
        params.length = Number(narrowLength)
        break
      case 'chunk':
        params.chunks = Number(chunkN)
        params.dim    = Number(chunkDim)
        break
      case 'cat': {
        let extras
        try { extras = JSON.parse(catTensors) }
        catch { alert('Invalid JSON for extra tensors'); return }
        // currentTensor is always prepended as tensors[0]
        params.tensors = [currentTensor, ...extras]
        params.dim     = Number(catDim)
        break
      }
      case 'stack': {
        let extras
        try { extras = JSON.parse(stackTensors) }
        catch { alert('Invalid JSON for extra tensors'); return }
        params.tensors = [currentTensor, ...extras]
        params.dim     = Number(stackDim)
        break
      }
      default:
        break
    }

    await applyOperation(selectedOp, params)
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Apply Operation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">

        {/* Operation selector */}
        <div className="space-y-1">
          <Label>Operation</Label>
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

        <Separator />

        {/* ── Binary: second tensor ─────────────────────────────────────── */}
        {isBinary && (
          <div className="space-y-1">
            <Label>Second tensor (JSON)</Label>
            <Textarea value={tensorB} onChange={(e) => setTensorB(e.target.value)}
              placeholder="e.g. [[1,2],[3,4]]" className="font-mono text-xs" />
          </div>
        )}

        {/* ── Clamp ────────────────────────────────────────────────────────── */}
        {selectedOp === 'clamp' && (
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label>Min (optional)</Label>
              <Input type="number" value={clampMin} onChange={(e) => setClampMin(e.target.value)} placeholder="−∞" />
            </div>
            <div className="space-y-1">
              <Label>Max (optional)</Label>
              <Input type="number" value={clampMax} onChange={(e) => setClampMax(e.target.value)} placeholder="+∞" />
            </div>
          </div>
        )}

        {/* ── Sum ──────────────────────────────────────────────────────────── */}
        {selectedOp === 'sum' && (
          <div className="space-y-3">
            <div className="space-y-1">
              <Label>Dimension (empty = all)</Label>
              <Input type="number" value={sumDim} onChange={(e) => setSumDim(e.target.value)} placeholder="optional" />
            </div>
            <div className="flex items-center gap-2">
              <Checkbox id="keepdim" checked={keepdim} onCheckedChange={setKeepdim} />
              <Label htmlFor="keepdim">Keep dimensions</Label>
            </div>
          </div>
        )}

        {/* ── Reshape ──────────────────────────────────────────────────────── */}
        {selectedOp === 'reshape' && (
          <div className="space-y-1">
            <Label>New shape (comma-separated)</Label>
            <Input value={reshapeShape} onChange={(e) => setReshapeShape(e.target.value)}
              placeholder="e.g. 1,9" className="font-mono" />
          </div>
        )}

        {/* ── Transpose ────────────────────────────────────────────────────── */}
        {selectedOp === 'transpose' && (
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label>dim0</Label>
              <Input type="number" value={dim0} onChange={(e) => setDim0(e.target.value)} />
            </div>
            <div className="space-y-1">
              <Label>dim1</Label>
              <Input type="number" value={dim1} onChange={(e) => setDim1(e.target.value)} />
            </div>
          </div>
        )}

        {/* ── Flatten ──────────────────────────────────────────────────────── */}
        {selectedOp === 'flatten' && (
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label>start_dim</Label>
              <Input type="number" value={flatStartDim} onChange={(e) => setFlatStartDim(e.target.value)} />
            </div>
            <div className="space-y-1">
              <Label>end_dim</Label>
              <Input type="number" value={flatEndDim} onChange={(e) => setFlatEndDim(e.target.value)} />
            </div>
          </div>
        )}

        {/* ── Squeeze ──────────────────────────────────────────────────────── */}
        {selectedOp === 'squeeze' && (
          <div className="space-y-1">
            <Label>dim (empty = all size-1 dims)</Label>
            <Input type="number" value={squeezeDim} onChange={(e) => setSqueezeDim(e.target.value)}
              placeholder="optional" />
          </div>
        )}

        {/* ── Unsqueeze ────────────────────────────────────────────────────── */}
        {selectedOp === 'unsqueeze' && (
          <div className="space-y-1">
            <Label>dim</Label>
            <Input type="number" value={unsqueezeDim} onChange={(e) => setUnsqueezeDim(e.target.value)} />
          </div>
        )}

        {/* ── Permute ──────────────────────────────────────────────────────── */}
        {selectedOp === 'permute' && (
          <div className="space-y-1">
            <Label>dims (comma-separated, e.g. 1,0 or 2,0,1)</Label>
            <Input value={permuteDims} onChange={(e) => setPermuteDims(e.target.value)}
              placeholder="1,0" className="font-mono" />
          </div>
        )}

        {/* ── Tile ─────────────────────────────────────────────────────────── */}
        {selectedOp === 'tile' && (
          <div className="space-y-1">
            <Label>dims — repeat count per axis (comma-separated)</Label>
            <Input value={tileDims} onChange={(e) => setTileDims(e.target.value)}
              placeholder="2,2" className="font-mono" />
          </div>
        )}

        {/* ── Repeat ───────────────────────────────────────────────────────── */}
        {selectedOp === 'repeat' && (
          <div className="space-y-1">
            <Label>sizes — copies per axis (comma-separated)</Label>
            <Input value={repeatSizes} onChange={(e) => setRepeatSizes(e.target.value)}
              placeholder="2,2" className="font-mono" />
          </div>
        )}

        {/* ── Narrow ───────────────────────────────────────────────────────── */}
        {selectedOp === 'narrow' && (
          <div className="grid grid-cols-3 gap-2">
            <div className="space-y-1">
              <Label>dim</Label>
              <Input type="number" value={narrowDim} onChange={(e) => setNarrowDim(e.target.value)} />
            </div>
            <div className="space-y-1">
              <Label>start</Label>
              <Input type="number" value={narrowStart} onChange={(e) => setNarrowStart(e.target.value)} />
            </div>
            <div className="space-y-1">
              <Label>length</Label>
              <Input type="number" value={narrowLength} onChange={(e) => setNarrowLength(e.target.value)} />
            </div>
          </div>
        )}

        {/* ── Chunk ────────────────────────────────────────────────────────── */}
        {selectedOp === 'chunk' && (
          <>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label>chunks</Label>
                <Input type="number" value={chunkN} onChange={(e) => setChunkN(e.target.value)} min={1} />
              </div>
              <div className="space-y-1">
                <Label>dim</Label>
                <Input type="number" value={chunkDim} onChange={(e) => setChunkDim(e.target.value)} />
              </div>
            </div>
            {chunkTensors && (
              <div className="rounded-md border border-dashed border-amber-300 bg-amber-50 p-2 text-xs text-amber-800 space-y-1">
                <p className="font-medium">{chunkTensors.length} chunks produced — current tensor = chunk[0]</p>
                {chunkTensors.map((c, i) => (
                  <p key={i} className="font-mono text-[11px]">
                    chunk[{i}] shape: [{c.shape.join(', ')}]
                  </p>
                ))}
              </div>
            )}
          </>
        )}

        {/* ── Cat ──────────────────────────────────────────────────────────── */}
        {selectedOp === 'cat' && (
          <div className="space-y-3">
            <div className="space-y-1">
              <Label>Extra tensors (JSON array, current tensor is always first)</Label>
              <Textarea value={catTensors} onChange={(e) => setCatTensors(e.target.value)}
                placeholder="e.g. [[[1,2],[3,4]]]" className="font-mono text-xs" rows={4} />
            </div>
            <div className="space-y-1">
              <Label>dim</Label>
              <Input type="number" value={catDim} onChange={(e) => setCatDim(e.target.value)} />
            </div>
          </div>
        )}

        {/* ── Stack ────────────────────────────────────────────────────────── */}
        {selectedOp === 'stack' && (
          <div className="space-y-3">
            <div className="space-y-1">
              <Label>Extra tensors (JSON array, all must match current tensor shape)</Label>
              <Textarea value={stackTensors} onChange={(e) => setStackTensors(e.target.value)}
                placeholder="e.g. [[[1,2],[3,4]]]" className="font-mono text-xs" rows={4} />
            </div>
            <div className="space-y-1">
              <Label>dim</Label>
              <Input type="number" value={stackDim} onChange={(e) => setStackDim(e.target.value)} />
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
