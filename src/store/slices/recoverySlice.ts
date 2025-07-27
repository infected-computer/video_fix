import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { RecoveredFile, ProgressInfo } from '@/types';

interface ScanOptions {
  fileTypes: string[];
  scanDepth: 'quick' | 'deep' | 'thorough';
  includeFragmented: boolean;
}

interface RecoveryState {
  selectedDrive: string | null;
  scanOptions: ScanOptions;
  scanResults: RecoveredFile[];
  selectedFiles: string[];
  recoveryProgress: ProgressInfo | null;
  isScanning: boolean;
  isRecovering: boolean;
  error: string | null;
  searchQuery: string;
  sortBy: 'name' | 'size' | 'confidence';
  sortOrder: 'asc' | 'desc';
}

const initialState: RecoveryState = {
  selectedDrive: null,
  scanOptions: {
    fileTypes: ['mp4', 'mov', 'avi', 'mkv'],
    scanDepth: 'quick',
    includeFragmented: false,
  },
  scanResults: [],
  selectedFiles: [],
  recoveryProgress: null,
  isScanning: false,
  isRecovering: false,
  error: null,
  searchQuery: '',
  sortBy: 'name',
  sortOrder: 'asc',
};

// Async thunks
export const startScan = createAsyncThunk(
  'recovery/startScan',
  async (_, { getState }) => {
    const state = getState() as any;
    const { selectedDrive, scanOptions } = state.recovery;
    
    if (!selectedDrive) {
      throw new Error('No drive selected for scanning');
    }
    
    const result = await window.electronAPI.python.startRecovery(
      selectedDrive,
      '', // Output directory not needed for scan
      { ...scanOptions, scanOnly: true }
    );
    return result;
  }
);

export const startRecovery = createAsyncThunk(
  'recovery/startRecovery',
  async (outputDirectory: string, { getState }) => {
    const state = getState() as any;
    const { selectedDrive, selectedFiles } = state.recovery;
    
    if (!selectedDrive || selectedFiles.length === 0) {
      throw new Error('No drive or files selected for recovery');
    }
    
    const result = await window.electronAPI.python.startRecovery(
      selectedDrive,
      outputDirectory,
      { selectedFiles }
    );
    return result;
  }
);

const recoverySlice = createSlice({
  name: 'recovery',
  initialState,
  reducers: {
    setSelectedDrive: (state, action: PayloadAction<string | null>) => {
      state.selectedDrive = action.payload;
      state.scanResults = []; // Clear previous results
      state.selectedFiles = [];
      state.error = null;
    },
    updateScanOptions: (state, action: PayloadAction<Partial<ScanOptions>>) => {
      state.scanOptions = { ...state.scanOptions, ...action.payload };
    },
    toggleFileSelection: (state, action: PayloadAction<string>) => {
      const fileId = action.payload;
      const index = state.selectedFiles.indexOf(fileId);
      if (index > -1) {
        state.selectedFiles.splice(index, 1);
      } else {
        state.selectedFiles.push(fileId);
      }
    },
    selectAllFiles: (state) => {
      state.selectedFiles = state.scanResults.map(file => file.id);
    },
    deselectAllFiles: (state) => {
      state.selectedFiles = [];
    },
    updateProgress: (state, action: PayloadAction<ProgressInfo>) => {
      state.recoveryProgress = action.payload;
    },
    setSearchQuery: (state, action: PayloadAction<string>) => {
      state.searchQuery = action.payload;
    },
    setSorting: (state, action: PayloadAction<{ sortBy: typeof initialState.sortBy; sortOrder: typeof initialState.sortOrder }>) => {
      state.sortBy = action.payload.sortBy;
      state.sortOrder = action.payload.sortOrder;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
    resetRecovery: (state) => {
      state.selectedDrive = null;
      state.scanResults = [];
      state.selectedFiles = [];
      state.recoveryProgress = null;
      state.isScanning = false;
      state.isRecovering = false;
      state.error = null;
      state.searchQuery = '';
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(startScan.pending, (state) => {
        state.isScanning = true;
        state.error = null;
        state.scanResults = [];
      })
      .addCase(startScan.fulfilled, (state, action) => {
        state.isScanning = false;
        state.scanResults = action.payload.recoveredFiles || [];
      })
      .addCase(startScan.rejected, (state, action) => {
        state.isScanning = false;
        state.error = action.error.message || 'Scan failed';
      })
      .addCase(startRecovery.pending, (state) => {
        state.isRecovering = true;
        state.error = null;
        state.recoveryProgress = {
          percentage: 0,
          status: 'Starting recovery...',
          estimatedTimeRemaining: 0,
          currentOperation: 'Initializing',
        };
      })
      .addCase(startRecovery.fulfilled, (state) => {
        state.isRecovering = false;
        state.recoveryProgress = {
          percentage: 100,
          status: 'Recovery completed',
          estimatedTimeRemaining: 0,
          currentOperation: 'Completed',
        };
      })
      .addCase(startRecovery.rejected, (state, action) => {
        state.isRecovering = false;
        state.error = action.error.message || 'Recovery failed';
        state.recoveryProgress = null;
      });
  },
});

export const {
  setSelectedDrive,
  updateScanOptions,
  toggleFileSelection,
  selectAllFiles,
  deselectAllFiles,
  updateProgress,
  setSearchQuery,
  setSorting,
  setError,
  clearError,
  resetRecovery,
} = recoverySlice.actions;

export default recoverySlice.reducer;