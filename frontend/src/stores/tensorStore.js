import { create } from 'zustand'
import api from '../../api/client'

export const useTensorStore = create((set, get) => ({
  originalTensor: null,
  currentTensor:  null,
  operations:     [],
  graphNodes:     null,   // GraphNode[] | null
  graphEdges:     null,   // GraphEdge[] | null
  history:        [],
  stats:          null,
  chunkTensors:   null,

  addHistory: (description) => {
    set((state) => ({
      history: [
        ...state.history,
        { time: new Date().toLocaleTimeString(), desc: description },
      ],
    }))
  },

  createTensor: async (op, shape) => {
    try {
      const response = await api.createTensor(op, shape)
      const tensor = response.data.data
      set({
        originalTensor: tensor,
        currentTensor:  tensor,
        operations:     [],
        graphNodes:     null,
        graphEdges:     null,
        stats:          null,
        chunkTensors:   null,
      })
      get().addHistory(`Created tensor with ${op} shape [${shape}]`)
      await get().fetchStats()
    } catch (error) {
      console.error('Creation failed:', error)
      throw error
    }
  },

  applyOperation: async (opName, params = {}) => {
    const { currentTensor } = get()
    if (!currentTensor) {
      alert('Please create a tensor first.')
      return
    }
    try {
      const response = await api.applyOperation(opName, currentTensor, params)
      const responseData = response.data

      const isChunk = opName === 'chunk'
      const resultTensor = isChunk
        ? responseData.tensors[0].data
        : responseData.data

      const opRecord = { op: opName, params, tensor_b: params.tensor_b ?? null }

      set((state) => ({
        operations:    [...state.operations, opRecord],
        currentTensor: resultTensor,
        chunkTensors:  isChunk ? responseData.tensors : null,
      }))

      const chunkLabel = isChunk
        ? ` (${responseData.tensors.length} chunks, following chunk[0])`
        : ''
      get().addHistory(`Applied ${opName}${chunkLabel}`)

      await Promise.all([get().fetchStats(), get().generateGraph()])
    } catch (error) {
      console.error('Operation failed:', error)
      alert('Operation failed: ' + (error.response?.data?.detail ?? error.message))
    }
  },

  fetchStats: async () => {
    const { currentTensor } = get()
    if (!currentTensor) {
      set({ stats: null })
      return
    }
    try {
      const response = await api.fetchStats(currentTensor)
      set({ stats: response.data })
    } catch (error) {
      console.error('Stats fetch failed:', error)
    }
  },

  generateGraph: async () => {
    const { originalTensor, operations } = get()
    if (!originalTensor || operations.length === 0) {
      set({ graphNodes: null, graphEdges: null })
      return
    }
    try {
      const response = await api.generateCumulativeGraph(originalTensor, operations)
      set({
        graphNodes: response.data.nodes,
        graphEdges: response.data.edges,
      })
    } catch (error) {
      console.error('Graph generation failed:', error)
    }
  },

  reset: () => {
    set({
      originalTensor: null,
      currentTensor:  null,
      operations:     [],
      graphNodes:     null,
      graphEdges:     null,
      history:        [],
      stats:          null,
      chunkTensors:   null,
    })
  },
}))
