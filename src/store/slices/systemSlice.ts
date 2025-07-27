import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { DriveInfo, SystemHealth, Operation } from '@/types';

interface SystemState {
  drives: DriveInfo[];
  systemHealth: SystemHealth;
  activeOperations: Operation[];
  isLoading: boolean;
  error: string | null;
}

const initialState: SystemState = {
  drives: [],
  systemHealth: {
    cpu: 0,
    memory: 0,
    disk: 0,
    status: 'good',
  },
  activeOperations: [],
  isLoading: false,
  error: null,
};

// Async thunks
export const fetchDrives = createAsyncThunk(
  'system/fetchDrives',
  async () => {
    const drives = await window.electronAPI.system.getDrives();
    return drives;
  }
);

export const refreshSystemHealth = createAsyncThunk(
  'system/refreshSystemHealth',
  async () => {
    // Mock system health data - would be replaced with real system monitoring
    return {
      cpu: Math.random() * 100,
      memory: Math.random() * 100,
      disk: Math.random() * 100,
      status: 'good' as const,
    };
  }
);

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    addOperation: (state, action: PayloadAction<Operation>) => {
      state.activeOperations.push(action.payload);
    },
    updateOperation: (state, action: PayloadAction<{ id: string; updates: Partial<Operation> }>) => {
      const { id, updates } = action.payload;
      const operation = state.activeOperations.find(op => op.id === id);
      if (operation) {
        Object.assign(operation, updates);
      }
    },
    removeOperation: (state, action: PayloadAction<string>) => {
      state.activeOperations = state.activeOperations.filter(op => op.id !== action.payload);
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchDrives.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchDrives.fulfilled, (state, action) => {
        state.isLoading = false;
        state.drives = action.payload;
      })
      .addCase(fetchDrives.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to fetch drives';
      })
      .addCase(refreshSystemHealth.fulfilled, (state, action) => {
        state.systemHealth = action.payload;
      });
  },
});

export const { addOperation, updateOperation, removeOperation, setError, clearError } = systemSlice.actions;
export default systemSlice.reducer;