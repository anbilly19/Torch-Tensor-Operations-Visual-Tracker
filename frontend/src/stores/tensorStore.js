import { create } from 'zustand'
import api from '../../api/client'

export const useTensorStore = create((set, get) => ({
  originalTensor: null,
  currentTensor: null,
  operations: [],
  graphImage: null,
  history: [],

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
        currentTensor: tensor,
        operations: [],
        graphImage: null,
      })
      get().addHistory(`Created tensor with ${op} shape [${shape}]`)
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
      const resultTensor = response.data.data
      const opRecord = { op: opName, params, tensor_b: params.tensor_b }
      set((state) => ({
        operations: [...state.operations, opRecord],
        currentTensor: resultTensor,
      }))
      get().addHistory(`Applied ${opName}`)
      await get().generateGraph()
    } catch (error) {
      console.error('Operation failed:', error)
      alert('Operation failed: ' + error.message)
    }
  },

  generateGraph: async () => {
    const { originalTensor, operations } = get()
    if (!originalTensor || operations.length === 0) {
      set({ graphImage: null })
      return
    }
    try {
      const response = await api.generateCumulativeGraph(originalTensor, operations)
      set({ graphImage: response.data.image })
    } catch (error) {
      console.error('Graph generation failed:', error)
    }
  },

  reset: () => {
    set({
      originalTensor: null,
      currentTensor: null,
      operations: [],
      graphImage: null,
      history: [],
    })
  },
}))
