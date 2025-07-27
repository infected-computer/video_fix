import { contextBridge, ipcRenderer } from 'electron';

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Python backend operations
  python: {
    startRepair: (filePath: string, options: any) => 
      ipcRenderer.invoke('python:start-repair', filePath, options),
    startRecovery: (sourcePath: string, outputDir: string, options: any) => 
      ipcRenderer.invoke('python:start-recovery', sourcePath, outputDir, options),
    analyzeFile: (filePath: string) => 
      ipcRenderer.invoke('python:analyze-file', filePath),
    cancel: () => 
      ipcRenderer.invoke('python:cancel'),
    
    // Event listeners for progress updates
    onProgress: (callback: (data: any) => void) => {
      ipcRenderer.on('python:progress', (event, data) => callback(data));
    },
    onLog: (callback: (message: string) => void) => {
      ipcRenderer.on('python:log', (event, message) => callback(message));
    },
    onError: (callback: (error: string) => void) => {
      ipcRenderer.on('python:error', (event, error) => callback(error));
    },
    
    // Remove listeners
    removeAllListeners: () => {
      ipcRenderer.removeAllListeners('python:progress');
      ipcRenderer.removeAllListeners('python:log');
      ipcRenderer.removeAllListeners('python:error');
    }
  },
  
  // File system operations
  fs: {
    selectFile: () => ipcRenderer.invoke('fs:select-file'),
    selectFolder: () => ipcRenderer.invoke('fs:select-folder'),
    saveFile: (defaultPath?: string) => ipcRenderer.invoke('fs:save-file', defaultPath)
  },
  
  // System information
  system: {
    getDrives: () => ipcRenderer.invoke('system:get-drives'),
    getInfo: () => ipcRenderer.invoke('system:get-info')
  }
});

// Type definitions for the exposed API
export interface ElectronAPI {
  python: {
    startRepair: (filePath: string, options: any) => Promise<any>;
    startRecovery: (sourcePath: string, outputDir: string, options: any) => Promise<any>;
    analyzeFile: (filePath: string) => Promise<any>;
    cancel: () => Promise<boolean>;
    onProgress: (callback: (data: any) => void) => void;
    onLog: (callback: (message: string) => void) => void;
    onError: (callback: (error: string) => void) => void;
    removeAllListeners: () => void;
  };
  fs: {
    selectFile: () => Promise<string | null>;
    selectFolder: () => Promise<string | null>;
    saveFile: (defaultPath?: string) => Promise<string | null>;
  };
  system: {
    getDrives: () => Promise<any[]>;
    getInfo: () => Promise<any>;
  };
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}