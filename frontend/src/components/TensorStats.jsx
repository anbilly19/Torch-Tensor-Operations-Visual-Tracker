import React from 'react'
import { useTensorStore } from '@/stores/tensorStore'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { BarChart2 } from 'lucide-react'

const STAT_FIELDS = [
  { key: 'mean',  label: 'Mean',       decimal: true  },
  { key: 'std',   label: 'Std Dev',    decimal: true  },
  { key: 'min',   label: 'Min',        decimal: true  },
  { key: 'max',   label: 'Max',        decimal: true  },
  { key: 'sum',   label: 'Sum',        decimal: true  },
  { key: 'norm',  label: 'L2 Norm',    decimal: true  },
  { key: 'numel', label: 'Elements',   decimal: false },
  { key: 'rank',  label: 'Rank (ndim)', decimal: false },
]

function fmt(value, decimal) {
  if (value === null || value === undefined) return '—'
  if (!decimal) return String(value)
  const abs = Math.abs(value)
  // Use exponential notation for very large or very small values
  if (abs !== 0 && (abs < 0.001 || abs >= 1e6)) return value.toExponential(3)
  return value.toFixed(4)
}

export default function TensorStats() {
  const stats = useTensorStore((s) => s.stats)

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <BarChart2 className="h-4 w-4" />
          Tensor Statistics
        </CardTitle>
      </CardHeader>
      <CardContent>
        {!stats ? (
          <p className="text-sm text-muted-foreground italic">
            No tensor yet. Create one to see statistics.
          </p>
        ) : (
          <div className="grid grid-cols-4 gap-2">
            {STAT_FIELDS.map(({ key, label, decimal }) => (
              <div
                key={key}
                className="flex flex-col items-center justify-center rounded-lg bg-zinc-50 border border-zinc-200 px-3 py-3 gap-1"
              >
                <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">
                  {label}
                </span>
                <span className="text-sm font-semibold font-mono text-foreground">
                  {fmt(stats[key], decimal)}
                </span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
