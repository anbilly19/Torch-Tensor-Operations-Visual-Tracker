import React, { useState } from 'react'
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
      { value: 'add',    label: 'Add',              binary: true  },
      { value: 'sub',    label: 'Subtract',         binary: true  },
      { value: 'mul',    label: 'Multiply',         binary: true  },
      { value: 'div',    label: 'Divide',           binary: true  },
      { value: 'matmul', label: 'Matrix Multiply',  binary: true  },
      { value: 'abs',    label: 'Absolute Value',   binary: false },
      { value: 'neg',    label: 'Negate',           binary: false },
      { value: 'clamp',  label: 'Clamp',            binary: false },
    ],
  },
  {
    group: 'Reduction',
    ops: [
      { value: 'sum',    label: 'Sum',              binary: false },
    ],
  },
  {
    group: 'Shape',
    ops: [
      { value: 'reshape',    label: 'Reshape',    binary: false },
      { value: 'transpose',  label: 'Transpose',  binary: false },
    ],
  },
]

const ALL_OPS = OPERATIONS.flatMap((g) => g.ops)

export default function OperationPanel() {
  const applyOperation = useTensorStore((s) => s.applyOperation)

  const [selectedOp, setSelectedOp] = useState('add')
  const [tensorB, setTensorB]       = useState('[[5,6],[7,8]]')
  const [sumDim, setSumDim]         = useState('')
  const [keepdim, setKeepdim]       = useState(false)
  const [reshapeShape, setReshapeShape] = useState('2,3')
  const [dim0, setDim0]             = useState('0')
  const [dim1, setDim1]             = useState('1')
  const [clampMin, setClampMin]     = useState('')
  const [clampMax, setClampMax]     = useState('')

  const isBinary = ALL_OPS.find((o) => o.value === selectedOp)?.binary ?? false

  async function handleApply() {
    let params = {}

    if (isBinary) {
      try {
        params.tensor_b = JSON.parse(tensorB)
      } catch {
        alert('Invalid JSON for second tensor')
        return
      }
    }

    if (selectedOp === 'clamp') {
      if (clampMin !== '') params.min_val = Number(clampMin)
      if (clampMax !== '') params.max_val = Number(clampMax)
    } else if (selectedOp === 'sum') {
      params.dim     = sumDim !== '' ? Number(sumDim) : null
      params.keepdim = keepdim
    } else if (selectedOp === 'reshape') {
      params.shape = reshapeShape.split(',').map(Number)
    } else if (selectedOp === 'transpose') {
      params.dim0 = Number(dim0)
      params.dim1 = Number(dim1)
    }

    await applyOperation(selectedOp, params)
  }

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
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {OPERATIONS.map((group) => (
                <SelectGroup key={group.group}>
                  <SelectLabel>{group.group}</SelectLabel>
                  {group.ops.map((op) => (
                    <SelectItem key={op.value} value={op.value}>
                      {op.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {/* Binary — second tensor */}
        {isBinary && (
          <div className="space-y-1">
            <Label>Second tensor (JSON list)</Label>
            <Textarea
              value={tensorB}
              onChange={(e) => setTensorB(e.target.value)}
              placeholder="e.g. [[1,2],[3,4]]"
              className="font-mono text-xs"
            />
          </div>
        )}

        {/* Clamp params */}
        {selectedOp === 'clamp' && (
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label>Min (optional)</Label>
              <Input
                type="number"
                value={clampMin}
                onChange={(e) => setClampMin(e.target.value)}
                placeholder="−∞"
              />
            </div>
            <div className="space-y-1">
              <Label>Max (optional)</Label>
              <Input
                type="number"
                value={clampMax}
                onChange={(e) => setClampMax(e.target.value)}
                placeholder="+∞"
              />
            </div>
          </div>
        )}

        {/* Sum params */}
        {selectedOp === 'sum' && (
          <div className="space-y-3">
            <div className="space-y-1">
              <Label>Dimension (leave empty for all)</Label>
              <Input
                type="number"
                value={sumDim}
                onChange={(e) => setSumDim(e.target.value)}
                placeholder="optional"
              />
            </div>
            <div className="flex items-center gap-2">
              <Checkbox id="keepdim" checked={keepdim} onCheckedChange={setKeepdim} />
              <Label htmlFor="keepdim">Keep dimensions</Label>
            </div>
          </div>
        )}

        {/* Reshape params */}
        {selectedOp === 'reshape' && (
          <div className="space-y-1">
            <Label>New shape (comma-separated)</Label>
            <Input
              value={reshapeShape}
              onChange={(e) => setReshapeShape(e.target.value)}
              placeholder="e.g. 2,3"
            />
          </div>
        )}

        {/* Transpose params */}
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

        <Button className="w-full" onClick={handleApply}>
          <Play className="mr-2 h-4 w-4" />
          Apply
        </Button>

      </CardContent>
    </Card>
  )
}
