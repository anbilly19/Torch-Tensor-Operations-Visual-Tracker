import React from 'react'
import { useTensorStore } from '@/stores/tensorStore'
import TensorCreator from '@/components/TensorCreator'
import OperationPanel from '@/components/OperationPanel'
import TensorDisplay from '@/components/TensorDisplay'
import TensorStats from '@/components/TensorStats'
import GraphDisplay from '@/components/GraphDisplay'
import HistoryList from '@/components/HistoryList'
import { Separator } from '@/components/ui/separator'
import { FlaskConical } from 'lucide-react'

export default function App() {
  const currentTensor = useTensorStore((s) => s.currentTensor)
  const graphNodes    = useTensorStore((s) => s.graphNodes)
  const graphEdges    = useTensorStore((s) => s.graphEdges)

  return (
    <div className="min-h-screen bg-zinc-50">
      {/* Header */}
      <header className="bg-white border-b border-zinc-200 px-6 py-4">
        <div className="max-w-screen-xl mx-auto flex items-center gap-3">
          <FlaskConical className="h-6 w-6 text-zinc-700" />
          <div>
            <h1 className="text-lg font-semibold tracking-tight">PyTorch Tensor Playground</h1>
            <p className="text-xs text-muted-foreground">
              Create a tensor, apply operations step by step, and watch the computation graph update.
            </p>
          </div>
        </div>
      </header>

      {/* Main layout */}
      <main className="max-w-screen-xl mx-auto px-6 py-6">
        <div className="flex gap-6">
          {/* Left panel */}
          <div className="w-72 shrink-0 flex flex-col gap-4">
            <TensorCreator />
            <OperationPanel />
            <HistoryList />
          </div>

          <Separator orientation="vertical" className="self-stretch" />

          {/* Right panel */}
          <div className="flex-1 flex flex-col gap-4 min-w-0">
            <TensorDisplay tensor={currentTensor} title="Current Tensor" />
            <TensorStats />
            <GraphDisplay graphNodes={graphNodes} graphEdges={graphEdges} />
          </div>
        </div>
      </main>
    </div>
  )
}
