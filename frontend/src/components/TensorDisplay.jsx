import React from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

function getShape(tensor) {
  if (!tensor) return null
  const dims = []
  let t = tensor
  while (Array.isArray(t)) { dims.push(t.length); t = t[0] }
  return dims
}

export default function TensorDisplay({ tensor, title, dtype }) {
  const shape = getShape(tensor)
  const shapeLabel = shape ? `[${shape.join(', ')}]` : null

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{title}</CardTitle>
          {shapeLabel && <Badge variant="secondary" className="font-mono">{shapeLabel}</Badge>}
        </div>
      </CardHeader>
      <CardContent>
        {!tensor ? (
          <p className="text-sm text-muted-foreground italic">No tensor yet.</p>
        ) : (
          <div className="flex flex-wrap gap-2 text-sm text-muted-foreground">
            <span>Rank <strong className="text-foreground">{shape.length}</strong></span>
            <span>·</span>
            <span>Numel <strong className="text-foreground">{shape.reduce((a, b) => a * b, 1)}</strong></span>
            {dtype && (
              <>
                <span>·</span>
                <span className="font-mono text-xs bg-zinc-100 border border-zinc-200 rounded px-1.5 py-0.5">{dtype}</span>
              </>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
