import React, { useState } from 'react'
import { useTensorStore } from '@/stores/tensorStore'
import TensorCreator from '@/components/TensorCreator'
import OperationPanel from '@/components/OperationPanel'
import TensorDisplay from '@/components/TensorDisplay'
import TensorStats from '@/components/TensorStats'
import GraphDisplay from '@/components/GraphDisplay'
import HistoryList from '@/components/HistoryList'
import { Separator } from '@/components/ui/separator'
import { FlaskConical, SlidersHorizontal, X } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function App() {
  const currentTensor = useTensorStore((s) => s.currentTensor)
  const graphNodes    = useTensorStore((s) => s.graphNodes)
  const graphEdges    = useTensorStore((s) => s.graphEdges)
  const dtype         = useTensorStore((s) => s.dtype)

  // Mobile: toggle left panel (controls) overlay
  const [panelOpen, setPanelOpen] = useState(false)

  return (
    <div className="min-h-screen bg-zinc-50 flex flex-col">

      {/* ── Header ────────────────────────────────────────────────────── */}
      <header className="bg-white border-b border-zinc-200 px-4 sm:px-6 py-3 sticky top-0 z-30">
        <div className="max-w-screen-xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">
            <FlaskConical className="h-5 w-5 sm:h-6 sm:w-6 text-zinc-700 shrink-0" />
            <div className="min-w-0">
              <h1 className="text-sm sm:text-lg font-semibold tracking-tight truncate">
                PyTorch Tensor Playground
              </h1>
              <p className="hidden sm:block text-xs text-muted-foreground">
                Create a tensor, apply operations step by step, and watch the computation graph update.
              </p>
            </div>
          </div>

          {/* Mobile: toggle controls panel */}
          <Button
            variant="outline"
            size="sm"
            className="lg:hidden shrink-0"
            onClick={() => setPanelOpen((v) => !v)}
          >
            {panelOpen
              ? <><X className="h-4 w-4 mr-1" />Close</>
              : <><SlidersHorizontal className="h-4 w-4 mr-1" />Controls</>}
          </Button>
        </div>
      </header>

      {/* ── Main ──────────────────────────────────────────────────────── */}
      <main className="flex-1 max-w-screen-xl w-full mx-auto px-4 sm:px-6 py-4 sm:py-6">

        {/* Desktop: side-by-side. Mobile: stacked. */}
        <div className="flex flex-col lg:flex-row gap-4 lg:gap-6">

          {/* ── Left panel (controls) ───────────────────────────────── */}
          {/*
            On mobile: hidden by default, shown as overlay when panelOpen.
            On lg+: always visible, fixed width.
          */}
          <div
            className={
              [
                // Mobile overlay
                'fixed inset-0 z-20 bg-black/40 lg:hidden transition-opacity duration-200',
                panelOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none',
              ].join(' ')
            }
            onClick={() => setPanelOpen(false)}
          />

          <aside
            className={
              [
                // Base
                'flex flex-col gap-4',
                // Mobile: slide-in drawer from left
                'fixed top-0 left-0 h-full w-80 max-w-[90vw] z-30 bg-zinc-50',
                'overflow-y-auto p-4 pt-16 transition-transform duration-200',
                panelOpen ? 'translate-x-0' : '-translate-x-full',
                // Desktop: static in flow
                'lg:static lg:translate-x-0 lg:h-auto lg:w-72 lg:shrink-0',
                'lg:overflow-visible lg:p-0 lg:pt-0 lg:bg-transparent lg:z-auto',
              ].join(' ')
            }
          >
            <TensorCreator />
            <OperationPanel />
            <HistoryList />
          </aside>

          {/* Desktop-only vertical divider */}
          <Separator orientation="vertical" className="self-stretch hidden lg:block" />

          {/* ── Right panel (output) ────────────────────────────────── */}
          <div className="flex-1 flex flex-col gap-4 min-w-0">
            <TensorDisplay tensor={currentTensor} title="Current Tensor" dtype={dtype} />
            <TensorStats />
            <GraphDisplay graphNodes={graphNodes} graphEdges={graphEdges} />
          </div>
        </div>
      </main>
    </div>
  )
}
