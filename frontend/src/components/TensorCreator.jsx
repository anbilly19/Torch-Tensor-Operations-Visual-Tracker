import React, { useState } from 'react'
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
import { PlusCircle } from 'lucide-react'

// Shape presets — chosen to demo every shape/indexing op cleanly
const SHAPE_PRESETS = [
  { label: '2×2',     value: '2,2',   hint: 'cat / stack / permute' },
  { label: '3×3',     value: '3,3',   hint: 'reshape / narrow / chunk' },
  { label: '4×3',     value: '4,3',   hint: 'flatten / tile / repeat' },
  { label: '2×1×2',   value: '2,1,2', hint: 'squeeze / unsqueeze' },
  { label: '1×3×3',   value: '1,3,3', hint: '3-D permute / flatten' },
]

export default function TensorCreator() {
  const createTensor = useTensorStore((s) => s.createTensor)
  const [op, setOp]             = useState('ones')
  const [shapeInput, setShapeInput] = useState('3,3')

  async function handleCreate() {
    const shape = shapeInput.split(',').map(Number)
    await createTensor(op, shape)
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Create Tensor</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">

        {/* Initializer */}
        <div className="space-y-1">
          <Label>Initializer</Label>
          <Select value={op} onValueChange={setOp}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ones">Ones</SelectItem>
              <SelectItem value="zeros">Zeros</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Shape input */}
        <div className="space-y-1">
          <Label>Shape (comma-separated)</Label>
          <Input
            value={shapeInput}
            onChange={(e) => setShapeInput(e.target.value)}
            placeholder="e.g. 3,3"
            className="font-mono"
          />
        </div>

        {/* Shape presets */}
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Quick presets</Label>
          <div className="flex flex-wrap gap-1.5">
            {SHAPE_PRESETS.map((p) => (
              <button
                key={p.value}
                type="button"
                title={p.hint}
                onClick={() => setShapeInput(p.value)}
                className={[
                  'rounded border px-2 py-0.5 text-xs font-mono transition-colors',
                  shapeInput === p.value
                    ? 'border-primary bg-primary text-primary-foreground'
                    : 'border-border bg-muted text-muted-foreground hover:border-primary hover:text-foreground',
                ].join(' ')}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <Button className="w-full" onClick={handleCreate}>
          <PlusCircle className="mr-2 h-4 w-4" />
          Create
        </Button>

      </CardContent>
    </Card>
  )
}
