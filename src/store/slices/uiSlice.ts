import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { UIState, ViewType } from '@/types';

const initialState: UIState = {
  currentView: 'dashboard',
  sidebarCollapsed: false,
  theme: 'light',
  language: 'en',
  isLoading: false,
  error: null,
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setCurrentView: (state, action: PayloadAction<ViewType>) => {
      state.currentView = action.payload;
    },
    toggleSidebar: (state) => {
      state.sidebarCollapsed = !state.sidebarCollapsed;
    },
    setSidebarCollapsed: (state, action: PayloadAction<boolean>) => {
      state.sidebarCollapsed = action.payload;
    },
    setTheme: (state, action: PayloadAction<'light' | 'dark'>) => {
      state.theme = action.payload;
    },
    setLanguage: (state, action: PayloadAction<'en' | 'he'>) => {
      state.language = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
});

export const {
  setCurrentView,
  toggleSidebar,
  setSidebarCollapsed,
  setTheme,
  setLanguage,
  setLoading,
  setError,
  clearError,
} = uiSlice.actions;

export default uiSlice.reducer;