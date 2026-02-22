import React from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

function format(val) {
  if (typeof val === 'number') return val.toFixed(3)
  return val
}

export default function TensorDisplay({ tensor, title }) {
  const shape = tensor
    ? `${tensor.length} × ${Array.isArray(tensor[0]) ? tensor[0].length : 1}`
    : null

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{title}</CardTitle>
          {shape && <Badge variant="secondary">{shape}</Badge>}
        </div>
      </CardHeader>
      <CardContent>
        {!tensor ? (
          <p className="text-sm text-muted-foreground italic">No tensor yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <div className="inline-block">
              {tensor.map((row, i) => (
                <div key={i} className="flex">
                  {(Array.isArray(row) ? row : [row]).map((val, j) => (
                    <span
                      key={j}
                      className="w-14 text-right px-2 py-1 m-px bg-zinc-50 border border-zinc-200 rounded text-xs font-mono"
                    >
                      {format(val)}
                    </span>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
