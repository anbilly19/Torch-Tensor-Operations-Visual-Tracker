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
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Play } from 'lucide-react'

export default function OperationPanel() {
  const applyOperation = useTensorStore((s) => s.applyOperation)
  const [selectedOp, setSelectedOp] = useState('add')
  const [tensorB, setTensorB] = useState('[[5,6],[7,8]]')
  const [sumDim, setSumDim] = useState('')
  const [keepdim, setKeepdim] = useState(false)
  const [reshapeShape, setReshapeShape] = useState('2,3')
  const [dim0, setDim0] = useState('0')
  const [dim1, setDim1] = useState('1')

  async function handleApply() {
    let params = {}
    if (selectedOp === 'add' || selectedOp === 'matmul') {
      try {
        params.tensor_b = JSON.parse(tensorB)
      } catch {
        alert('Invalid JSON for second tensor')
        return
      }
    } else if (selectedOp === 'sum') {
      params.dim = sumDim !== '' ? Number(sumDim) : null
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
        <div className="space-y-1">
          <Label>Operation</Label>
          <Select value={selectedOp} onValueChange={setSelectedOp}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="add">Add (requires second tensor)</SelectItem>
              <SelectItem value="matmul">Matrix Multiply (requires second tensor)</SelectItem>
              <SelectItem value="sum">Sum (optional dimension)</SelectItem>
              <SelectItem value="reshape">Reshape</SelectItem>
              <SelectItem value="transpose">Transpose</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Separator />

        {(selectedOp === 'add' || selectedOp === 'matmul') && (
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
              <Checkbox
                id="keepdim"
                checked={keepdim}
                onCheckedChange={setKeepdim}
              />
              <Label htmlFor="keepdim">Keep dimensions</Label>
            </div>
          </div>
        )}

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
