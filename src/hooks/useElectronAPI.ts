import { useEffect, useCallback } from 'react';
import { useAppDispatch } from './redux';
import { updateProgress as updateRepairProgress } from '@/store/slices/repairSlice';
import { updateProgress as updateRecoveryProgress } from '@/store/slices/recoverySlice';

export const useElectronAPI = () => {
  const dispatch = useAppDispatch();

  // Set up Python backend event listeners
  useEffect(() => {
    const handleProgress = (data: any) => {
      // Determine which slice to update based on operation type
      if (data.operationType === 'repair') {
        dispatch(updateRepairProgress(data));
      } else if (data.operationType === 'recovery') {
        dispatch(updateRecoveryProgress(data));
      }
    };

    const handleLog = (message: string) => {
      console.log('Python Backend:', message);
    };

    const handleError = (error: string) => {
      console.error('Python Backend Error:', error);
    };

    // Set up listeners
    window.electronAPI.python.onProgress(handleProgress);
    window.electronAPI.python.onLog(handleLog);
    window.electronAPI.python.onError(handleError);

    // Cleanup on unmount
    return () => {
      window.electronAPI.python.removeAllListeners();
    };
  }, [dispatch]);

  // File system operations
  const selectFile = useCallback(async () => {
    try {
      const filePath = await window.electronAPI.fs.selectFile();
      return filePath;
    } catch (error) {
      console.error('Failed to select file:', error);
      return null;
    }
  }, []);

  const selectFolder = useCallback(async () => {
    try {
      const folderPath = await window.electronAPI.fs.selectFolder();
      return folderPath;
    } catch (error) {
      console.error('Failed to select folder:', error);
      return null;
    }
  }, []);

  const saveFile = useCallback(async (defaultPath?: string) => {
    try {
      const filePath = await window.electronAPI.fs.saveFile(defaultPath);
      return filePath;
    } catch (error) {
      console.error('Failed to save file:', error);
      return null;
    }
  }, []);

  // System operations
  const getDrives = useCallback(async () => {
    try {
      const drives = await window.electronAPI.system.getDrives();
      return drives;
    } catch (error) {
      console.error('Failed to get drives:', error);
      return [];
    }
  }, []);

  const getSystemInfo = useCallback(async () => {
    try {
      const info = await window.electronAPI.system.getInfo();
      return info;
    } catch (error) {
      console.error('Failed to get system info:', error);
      return null;
    }
  }, []);

  // Python backend operations
  const startRepair = useCallback(async (filePath: string, options: any) => {
    try {
      const result = await window.electronAPI.python.startRepair(filePath, options);
      return result;
    } catch (error) {
      console.error('Failed to start repair:', error);
      throw error;
    }
  }, []);

  const startRecovery = useCallback(async (sourcePath: string, outputDir: string, options: any) => {
    try {
      const result = await window.electronAPI.python.startRecovery(sourcePath, outputDir, options);
      return result;
    } catch (error) {
      console.error('Failed to start recovery:', error);
      throw error;
    }
  }, []);

  const analyzeFile = useCallback(async (filePath: string) => {
    try {
      const result = await window.electronAPI.python.analyzeFile(filePath);
      return result;
    } catch (error) {
      console.error('Failed to analyze file:', error);
      throw error;
    }
  }, []);

  const cancelOperation = useCallback(async () => {
    try {
      const result = await window.electronAPI.python.cancel();
      return result;
    } catch (error) {
      console.error('Failed to cancel operation:', error);
      return false;
    }
  }, []);

  return {
    // File system
    selectFile,
    selectFolder,
    saveFile,
    
    // System
    getDrives,
    getSystemInfo,
    
    // Python backend
    startRepair,
    startRecovery,
    analyzeFile,
    cancelOperation,
  };
};