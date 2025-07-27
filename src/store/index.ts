import { configureStore } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';

import appSlice from './slices/appSlice';
import uiSlice from './slices/uiSlice';
import systemSlice from './slices/systemSlice';
import repairSlice from './slices/repairSlice';
import recoverySlice from './slices/recoverySlice';
import analysisSlice from './slices/analysisSlice';
import settingsSlice from './slices/settingsSlice';

export const store = configureStore({
  reducer: {
    app: appSlice,
    ui: uiSlice,
    system: systemSlice,
    repair: repairSlice,
    recovery: recoverySlice,
    analysis: analysisSlice,
    settings: settingsSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Typed hooks
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;