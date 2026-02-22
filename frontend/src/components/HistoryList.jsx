import React from 'react'
import { useTensorStore } from '@/stores/tensorStore'
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { RotateCcw, Clock } from 'lucide-react'

export default function HistoryList() {
  const history = useTensorStore((s) => s.history)
  const reset = useTensorStore((s) => s.reset)

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Clock className="h-4 w-4" />
          History
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-48 px-6">
          {history.length === 0 ? (
            <p className="text-sm text-muted-foreground italic py-4">No operations yet.</p>
          ) : (
            <div className="space-y-1 py-2">
              {history.map((item, idx) => (
                <div key={idx}>
                  <div className="flex items-start gap-2 py-1.5">
                    <span className="text-xs text-muted-foreground font-mono shrink-0">{item.time}</span>
                    <span className="text-xs">{item.desc}</span>
                  </div>
                  {idx < history.length - 1 && <Separator />}
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
      <CardFooter className="pt-3">
        <Button variant="outline" size="sm" className="w-full" onClick={reset}>
          <RotateCcw className="mr-2 h-3 w-3" />
          Reset All
        </Button>
      </CardFooter>
    </Card>
  )
}
