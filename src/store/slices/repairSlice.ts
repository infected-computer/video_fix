import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { FileInfo, ProgressInfo, RepairResult } from '@/types';

interface RepairOptions {
  aiEnhancement: boolean;
  outputPath: string;
  preserveOriginal: boolean;
}

interface RepairState {
  selectedFile: FileInfo | null;
  options: RepairOptions;
  progress: ProgressInfo | null;
  results: RepairResult | null;
  isRepairing: boolean;
  error: string | null;
}

const initialState: RepairState = {
  selectedFile: null,
  options: {
    aiEnhancement: false,
    outputPath: '',
    preserveOriginal: true,
  },
  progress: null,
  results: null,
  isRepairing: false,
  error: null,
};

// Async thunks
export const startRepair = createAsyncThunk(
  'repair/startRepair',
  async (_, { getState }) => {
    const state = getState() as any;
    const { selectedFile, options } = state.repair;
    
    if (!selectedFile) {
      throw new Error('No file selected for repair');
    }
    
    const result = await window.electronAPI.python.startRepair(selectedFile.path, options);
    return result;
  }
);

export const cancelRepair = createAsyncThunk(
  'repair/cancelRepair',
  async () => {
    const result = await window.electronAPI.python.cancel();
    return result;
  }
);

const repairSlice = createSlice({
  name: 'repair',
  initialState,
  reducers: {
    setSelectedFile: (state, action: PayloadAction<FileInfo | null>) => {
      state.selectedFile = action.payload;
      state.results = null; // Clear previous results
      state.error = null;
    },
    updateOptions: (state, action: PayloadAction<Partial<RepairOptions>>) => {
      state.options = { ...state.options, ...action.payload };
    },
    updateProgress: (state, action: PayloadAction<ProgressInfo>) => {
      state.progress = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
    resetRepair: (state) => {
      state.selectedFile = null;
      state.progress = null;
      state.results = null;
      state.isRepairing = false;
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(startRepair.pending, (state) => {
        state.isRepairing = true;
        state.error = null;
        state.progress = {
          percentage: 0,
          status: 'Starting repair...',
          estimatedTimeRemaining: 0,
          currentOperation: 'Initializing',
        };
      })
      .addCase(startRepair.fulfilled, (state, action) => {
        state.isRepairing = false;
        state.results = action.payload;
        state.progress = {
          percentage: 100,
          status: 'Repair completed',
          estimatedTimeRemaining: 0,
          currentOperation: 'Completed',
        };
      })
      .addCase(startRepair.rejected, (state, action) => {
        state.isRepairing = false;
        state.error = action.error.message || 'Repair failed';
        state.progress = null;
      })
      .addCase(cancelRepair.fulfilled, (state) => {
        state.isRepairing = false;
        state.progress = null;
        state.error = 'Repair cancelled by user';
      });
  },
});

export const {
  setSelectedFile,
  updateOptions,
  updateProgress,
  setError,
  clearError,
  resetRepair,
} = repairSlice.actions;

export default repairSlice.reducer;