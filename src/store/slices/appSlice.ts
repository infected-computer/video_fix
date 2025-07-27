import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

interface AppState {
  isInitialized: boolean;
  isLoading: boolean;
  error: string | null;
  version: string;
  platform: string;
}

const initialState: AppState = {
  isInitialized: false,
  isLoading: false,
  error: null,
  version: '2.0.0',
  platform: 'unknown',
};

// Async thunks
export const initializeApp = createAsyncThunk(
  'app/initialize',
  async () => {
    try {
      const systemInfo = await window.electronAPI.system.getInfo();
      return {
        platform: systemInfo.platform,
        version: systemInfo.version,
      };
    } catch (error) {
      throw new Error('Failed to initialize application');
    }
  }
);

const appSlice = createSlice({
  name: 'app',
  initialState,
  reducers: {
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(initializeApp.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(initializeApp.fulfilled, (state, action) => {
        state.isLoading = false;
        state.isInitialized = true;
        state.platform = action.payload.platform;
        state.version = action.payload.version;
      })
      .addCase(initializeApp.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Failed to initialize application';
      });
  },
});

export const { setError, clearError } = appSlice.actions;
export default appSlice.reducer;