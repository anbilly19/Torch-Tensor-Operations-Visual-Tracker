import { create } from 'zustand'
import api from '../../api/client'

export const useTensorStore = create((set, get) => ({
  originalTensor: null,
  currentTensor:  null,
  dtype:          null,
  operations:     [],
  graphNodes:     null,
  graphEdges:     null,
  history:        [],
  stats:          null,
  chunkTensors:   null,

  // Import-specific state
  importKeys:        [],   // keys returned when file is a dict
  importPendingFile: null, // File object held while user picks a key

  addHistory: (description) => {
    set((state) => ({
      history: [
        ...state.history,
        { time: new Date().toLocaleTimeString(), desc: description },
      ],
    }))
  },

  // ── Internal: set tensor from any source ────────────────────────────────
  _setTensor: (data, shape, dtype, historyLabel) => {
    set({
      originalTensor:    data,
      currentTensor:     data,
      dtype,
      operations:        [],
      graphNodes:        null,
      graphEdges:        null,
      stats:             null,
      chunkTensors:      null,
      importKeys:        [],
      importPendingFile: null,
    })
    get().addHistory(historyLabel)
    get().fetchStats()
  },

  // ── Create (ones / zeros) ────────────────────────────────────────────
  createTensor: async (op, shape) => {
    try {
      const res = await api.createTensor(op, shape)
      const d   = res.data
      get()._setTensor(d.data, d.shape, d.dtype, `Created ${op} [${shape}]`)
    } catch (error) {
      console.error('Creation failed:', error)
      throw error
    }
  },

  // ── Import: first upload ───────────────────────────────────────────────
  importTensor: async (file) => {
    try {
      const res = await api.importTensor(file)
      const d   = res.data

      if (d.data !== null && d.data !== undefined) {
        // Single tensor — load immediately
        const label = d.key_used
          ? `Imported "${d.key_used}" from ${file.name} [${d.shape}]`
          : `Imported ${file.name} [${d.shape}]`
        get()._setTensor(d.data, d.shape, d.dtype, label)
        return { needsKey: false }
      }

      // Dict file — surface keys to the UI
      set({ importKeys: d.keys, importPendingFile: file })
      return { needsKey: true, keys: d.keys }
    } catch (error) {
      console.error('Import failed:', error)
      const detail = error.response?.data?.detail ?? error.message
      throw new Error(detail)
    }
  },

  // ── Import: key selection (second upload) ───────────────────────────
  importTensorWithKey: async (key) => {
    const file = get().importPendingFile
    if (!file) throw new Error('No pending file — please re-upload.')
    try {
      const res = await api.importTensorWithKey(file, key)
      const d   = res.data
      get()._setTensor(
        d.data, d.shape, d.dtype,
        `Imported "${key}" from ${file.name} [${d.shape}]`,
      )
    } catch (error) {
      console.error('Key import failed:', error)
      const detail = error.response?.data?.detail ?? error.message
      throw new Error(detail)
    }
  },

  // ── Apply operation ──────────────────────────────────────────────────
  applyOperation: async (opName, params = {}) => {
    const { currentTensor } = get()
    if (!currentTensor) { alert('Please create or import a tensor first.'); return }
    try {
      const response    = await api.applyOperation(opName, currentTensor, params)
      const responseData = response.data
      const isChunk      = opName === 'chunk'
      const resultTensor = isChunk ? responseData.tensors[0].data : responseData.data
      const opRecord     = { op: opName, params, tensor_b: params.tensor_b ?? null }

      set((state) => ({
        operations:    [...state.operations, opRecord],
        currentTensor: resultTensor,
        chunkTensors:  isChunk ? responseData.tensors : null,
      }))

      const chunkLabel = isChunk
        ? ` (${responseData.tensors.length} chunks, following chunk[0])` : ''
      get().addHistory(`Applied ${opName}${chunkLabel}`)
      await Promise.all([get().fetchStats(), get().generateGraph()])
    } catch (error) {
      console.error('Operation failed:', error)
      alert('Operation failed: ' + (error.response?.data?.detail ?? error.message))
    }
  },

  // ── Stats ────────────────────────────────────────────────────────────────
  fetchStats: async () => {
    const { currentTensor } = get()
    if (!currentTensor) { set({ stats: null }); return }
    try {
      const response = await api.fetchStats(currentTensor)
      set({ stats: response.data })
    } catch (error) {
      console.error('Stats fetch failed:', error)
    }
  },

  // ── Graph ────────────────────────────────────────────────────────────────
  generateGraph: async () => {
    const { originalTensor, operations } = get()
    if (!originalTensor || operations.length === 0) {
      set({ graphNodes: null, graphEdges: null }); return
    }
    try {
      const response = await api.generateCumulativeGraph(originalTensor, operations)
      set({ graphNodes: response.data.nodes, graphEdges: response.data.edges })
    } catch (error) {
      console.error('Graph generation failed:', error)
    }
  },

  // ── Reset ────────────────────────────────────────────────────────────────
  reset: () => {
    set({
      originalTensor: null, currentTensor: null, dtype: null,
      operations: [], graphNodes: null, graphEdges: null,
      history: [], stats: null, chunkTensors: null,
      importKeys: [], importPendingFile: null,
    })
  },
}))
