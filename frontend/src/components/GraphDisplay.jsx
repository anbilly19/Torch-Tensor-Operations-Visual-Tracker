import React from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { GitGraph } from 'lucide-react'

export default function GraphDisplay({ image }) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <GitGraph className="h-4 w-4" />
          Computation Graph
        </CardTitle>
      </CardHeader>
      <CardContent>
        {!image ? (
          <div className="flex items-center justify-center h-48 rounded-lg border-2 border-dashed border-zinc-200">
            <p className="text-sm text-muted-foreground italic">
              No graph yet. Apply an operation.
            </p>
          </div>
        ) : (
          <img
            src={image}
            alt="Computation graph"
            className="max-w-full rounded-lg border border-zinc-200"
          />
        )}
      </CardContent>
    </Card>
  )
}
