import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { UserSettings } from '@/types';

const initialState: UserSettings = {
  language: 'en',
  theme: 'light',
  autoSave: true,
  cpuLimit: 80,
  memoryLimit: 4096,
  parallelProcessing: true,
  workerCount: 4,
  loggingLevel: 'info',
  tempDirectory: '',
  backendPreference: 'auto',
};

const settingsSlice = createSlice({
  name: 'settings',
  initialState,
  reducers: {
    updateSettings: (state, action: PayloadAction<Partial<UserSettings>>) => {
      return { ...state, ...action.payload };
    },
    setLanguage: (state, action: PayloadAction<'en' | 'he'>) => {
      state.language = action.payload;
    },
    setTheme: (state, action: PayloadAction<'light' | 'dark'>) => {
      state.theme = action.payload;
    },
    setAutoSave: (state, action: PayloadAction<boolean>) => {
      state.autoSave = action.payload;
    },
    setCpuLimit: (state, action: PayloadAction<number>) => {
      state.cpuLimit = Math.max(10, Math.min(100, action.payload));
    },
    setMemoryLimit: (state, action: PayloadAction<number>) => {
      state.memoryLimit = Math.max(512, action.payload);
    },
    setParallelProcessing: (state, action: PayloadAction<boolean>) => {
      state.parallelProcessing = action.payload;
    },
    setWorkerCount: (state, action: PayloadAction<number>) => {
      state.workerCount = Math.max(1, Math.min(16, action.payload));
    },
    setLoggingLevel: (state, action: PayloadAction<UserSettings['loggingLevel']>) => {
      state.loggingLevel = action.payload;
    },
    setTempDirectory: (state, action: PayloadAction<string>) => {
      state.tempDirectory = action.payload;
    },
    setBackendPreference: (state, action: PayloadAction<UserSettings['backendPreference']>) => {
      state.backendPreference = action.payload;
    },
    resetToDefaults: () => {
      return initialState;
    },
  },
});

export const {
  updateSettings,
  setLanguage,
  setTheme,
  setAutoSave,
  setCpuLimit,
  setMemoryLimit,
  setParallelProcessing,
  setWorkerCount,
  setLoggingLevel,
  setTempDirectory,
  setBackendPreference,
  resetToDefaults,
} = settingsSlice.actions;

export default settingsSlice.reducer;