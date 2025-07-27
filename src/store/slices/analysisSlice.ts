import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { FileInfo, AnalysisResult } from '@/types';

interface AnalysisState {
  selectedFile: FileInfo | null;
  analysisResults: AnalysisResult | null;
  isAnalyzing: boolean;
  error: string | null;
}

const initialState: AnalysisState = {
  selectedFile: null,
  analysisResults: null,
  isAnalyzing: false,
  error: null,
};

// Async thunks
export const startAnalysis = createAsyncThunk(
  'analysis/startAnalysis',
  async (filePath: string) => {
    const result = await window.electronAPI.python.analyzeFile(filePath);
    return result;
  }
);

export const exportAnalysisReport = createAsyncThunk(
  'analysis/exportReport',
  async (format: 'json' | 'pdf' | 'html', { getState }) => {
    const state = getState() as any;
    const { analysisResults } = state.analysis;
    
    if (!analysisResults) {
      throw new Error('No analysis results to export');
    }
    
    const defaultPath = `analysis_report_${Date.now()}.${format}`;
    const filePath = await window.electronAPI.fs.saveFile(defaultPath);
    
    if (!filePath) {
      throw new Error('Export cancelled by user');
    }
    
    // Here you would implement the actual export logic
    // For now, just return success
    return { filePath, format };
  }
);

const analysisSlice = createSlice({
  name: 'analysis',
  initialState,
  reducers: {
    setSelectedFile: (state, action: PayloadAction<FileInfo | null>) => {
      state.selectedFile = action.payload;
      state.analysisResults = null; // Clear previous results
      state.error = null;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
    resetAnalysis: (state) => {
      state.selectedFile = null;
      state.analysisResults = null;
      state.isAnalyzing = false;
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(startAnalysis.pending, (state) => {
        state.isAnalyzing = true;
        state.error = null;
      })
      .addCase(startAnalysis.fulfilled, (state, action) => {
        state.isAnalyzing = false;
        state.analysisResults = action.payload;
      })
      .addCase(startAnalysis.rejected, (state, action) => {
        state.isAnalyzing = false;
        state.error = action.error.message || 'Analysis failed';
      })
      .addCase(exportAnalysisReport.fulfilled, (state) => {
        // Could show a success message here
      })
      .addCase(exportAnalysisReport.rejected, (state, action) => {
        state.error = action.error.message || 'Export failed';
      });
  },
});

export const {
  setSelectedFile,
  setError,
  clearError,
  resetAnalysis,
} = analysisSlice.actions;

export default analysisSlice.reducer;