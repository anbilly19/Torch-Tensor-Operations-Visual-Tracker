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

export default function TensorCreator() {
  const createTensor = useTensorStore((s) => s.createTensor)
  const [op, setOp] = useState('ones')
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
        <div className="space-y-1">
          <Label>Shape (comma-separated)</Label>
          <Input
            value={shapeInput}
            onChange={(e) => setShapeInput(e.target.value)}
            placeholder="e.g. 3,3"
          />
        </div>
        <Button className="w-full" onClick={handleCreate}>
          <PlusCircle className="mr-2 h-4 w-4" />
          Create
        </Button>
      </CardContent>
    </Card>
  )
}
