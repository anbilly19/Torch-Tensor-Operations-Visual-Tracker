import React, { useMemo, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
  MarkerType,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { GitGraph, ExternalLink } from 'lucide-react'

// ── Stats popover ─────────────────────────────────────────────────────────────────
function fmt(v) {
  if (v === null || v === undefined) return '—'
  return typeof v === 'number' ? v.toFixed(4) : String(v)
}

function StatsPopover({ stats, shape }) {
  if (!stats) return null
  const rows = [
    ['shape', `[${shape?.join(', ')}]`],
    ['mean',  fmt(stats.mean)],
    ['std',   fmt(stats.std)],
    ['min',   fmt(stats.min)],
    ['max',   fmt(stats.max)],
    ['sum',   fmt(stats.sum)],
    ['norm',  fmt(stats.norm)],
    ['numel', stats.numel],
    ['rank',  stats.rank],
  ]
  return (
    <div className="absolute z-50 bottom-[calc(100%+8px)] left-1/2 -translate-x-1/2
                    w-44 rounded-lg border border-zinc-200 bg-white shadow-lg
                    p-2 text-[11px] pointer-events-none">
      <p className="font-semibold text-zinc-700 mb-1">Tensor stats</p>
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-2">
          <span className="text-zinc-400">{k}</span>
          <span className="font-mono text-zinc-700">{v}</span>
        </div>
      ))}
      <div className="absolute top-full left-1/2 -translate-x-1/2
                      border-4 border-transparent border-t-zinc-200" />
    </div>
  )
}

// ── Custom nodes ─────────────────────────────────────────────────────────────────
function TensorNode({ data }) {
  const [hovered, setHovered] = useState(false)
  return (
    <div className="relative cursor-default"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}>
      {hovered && <StatsPopover stats={data.stats} shape={data.shape} />}
      <Handle type="target" position={Position.Top} className="!bg-blue-400" />
      <div className="rounded-lg border-2 border-blue-300 bg-blue-50 px-3 py-2 min-w-[110px] text-center shadow-sm">
        <p className="font-semibold text-blue-800 text-[12px]">{data.label}</p>
        <p className="font-mono text-blue-500 text-[11px] mt-0.5">[{data.shape?.join(' × ')}]</p>
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-blue-400" />
    </div>
  )
}

function TensorBNode({ data }) {
  const [hovered, setHovered] = useState(false)
  return (
    <div className="relative cursor-default"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}>
      {hovered && <StatsPopover stats={data.stats} shape={data.shape} />}
      <div className="rounded-lg border-2 border-amber-300 bg-amber-50 px-3 py-2 min-w-[110px] text-center shadow-sm">
        <p className="font-semibold text-amber-800 text-[12px]">{data.label}</p>
        <p className="font-mono text-amber-500 text-[11px] mt-0.5">[{data.shape?.join(' × ')}]</p>
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-amber-400" />
    </div>
  )
}

function OpNode({ data }) {
  return (
    <div>
      <Handle type="target" position={Position.Top} className="!bg-emerald-400" />
      <div className="rounded-lg border-2 border-emerald-300 bg-emerald-50 px-3 py-2 min-w-[110px] text-center shadow-sm">
        <p className="font-semibold text-emerald-800 text-[12px]">{data.label}</p>
        {data.docs_url && (
          <a href={data.docs_url} target="_blank" rel="noopener noreferrer"
            className="inline-flex items-center gap-0.5 mt-1 text-[10px] text-blue-500 hover:text-blue-700 hover:underline"
            onClick={(e) => e.stopPropagation()}>
            <ExternalLink className="h-2.5 w-2.5" />docs
          </a>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-emerald-400" />
    </div>
  )
}

const NODE_TYPES = { tensor: TensorNode, tensor_b: TensorBNode, op: OpNode }

// ── Layout ───────────────────────────────────────────────────────────────────────
const H_GAP = 160
const V_GAP = 100

function layoutNodes(rawNodes) {
  const positioned = []
  let mainY = 0

  rawNodes.forEach((n) => {
    if (n.type === 'tensor') {
      positioned.push({ ...n, position: { x: 0, y: mainY } })
      mainY += V_GAP
    } else if (n.type === 'op') {
      positioned.push({ ...n, position: { x: 0, y: mainY } })
      mainY += V_GAP
    } else if (n.type === 'tensor_b') {
      positioned.push({ ...n, position: { x: H_GAP, y: mainY - V_GAP } })
    }
  })
  return positioned
}

function toFlowNodes(rawNodes) {
  return layoutNodes(rawNodes).map((n) => ({
    id: n.id, type: n.type, position: n.position,
    data: { label: n.label, shape: n.shape, stats: n.stats, docs_url: n.docs_url },
  }))
}

function toFlowEdges(rawEdges) {
  return rawEdges.map((e) => ({
    id: e.id, source: e.source, target: e.target, animated: true,
    markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16, color: '#94a3b8' },
    style:     { stroke: '#94a3b8', strokeWidth: 1.5 },
  }))
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function GraphDisplay({ graphNodes, graphEdges }) {
  const rfNodes = useMemo(() => toFlowNodes(graphNodes ?? []), [graphNodes])
  const rfEdges = useMemo(() => toFlowEdges(graphEdges ?? []), [graphEdges])
  const hasGraph = graphNodes && graphNodes.length > 0

  return (
    <Card className="flex flex-col">
      <CardHeader className="pb-3 shrink-0">
        <CardTitle className="text-base flex items-center gap-2 flex-wrap">
          <GitGraph className="h-4 w-4" />
          Computation Graph
          {hasGraph && (
            <span className="ml-auto text-xs font-normal text-muted-foreground hidden sm:inline">
              Hover nodes for stats · Drag to pan · Scroll to zoom
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0 overflow-hidden">
        {!hasGraph ? (
          <div className="flex items-center justify-center h-48 rounded-b-lg border-t border-zinc-100">
            <p className="text-sm text-muted-foreground italic">No graph yet. Apply an operation.</p>
          </div>
        ) : (
          // Responsive height: shorter on mobile, taller on larger screens
          <div className="w-full h-[320px] sm:h-[420px] lg:h-[480px]">
            <ReactFlow
              nodes={rfNodes}
              edges={rfEdges}
              nodeTypes={NODE_TYPES}
              fitView
              fitViewOptions={{ padding: 0.25 }}
              minZoom={0.2}
              maxZoom={2}
              proOptions={{ hideAttribution: true }}
            >
              <Background color="#e2e8f0" gap={16} />
              <Controls showInteractive={false} />
              <MiniMap
                nodeColor={(n) =>
                  n.type === 'tensor'   ? '#bfdbfe' :
                  n.type === 'tensor_b' ? '#fde68a' : '#bbf7d0'
                }
                maskColor="rgba(241,245,249,0.7)"
                className="!hidden sm:!block"
                style={{ bottom: 8, right: 8 }}
              />
            </ReactFlow>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
