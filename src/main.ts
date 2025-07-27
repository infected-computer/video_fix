import { app, BrowserWindow, ipcMain, dialog, shell } from 'electron';
import * as path from 'path';
import { spawn, ChildProcess } from 'child_process';

// Keep a global reference of the window object
let mainWindow: BrowserWindow | null = null;
let pythonProcess: ChildProcess | null = null;

const isDevelopment = process.env.NODE_ENV === 'development';

function createWindow(): void {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 1024,
    minHeight: 768,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    titleBarStyle: 'default',
    show: false, // Don't show until ready
    icon: path.join(__dirname, '../assets/icon.png'), // Add app icon
  });

  // Load the app
  if (isDevelopment) {
    mainWindow.loadURL('http://localhost:3000');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, 'index.html'));
  }

  // Show window when ready to prevent visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
    
    // Focus on window
    if (isDevelopment) {
      mainWindow?.webContents.openDevTools();
    }
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

// App event handlers
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    // On macOS, re-create window when dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  // On macOS, keep app running even when all windows are closed
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  // Clean up Python process
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
});

// IPC handlers for Python backend communication
ipcMain.handle('python:start-repair', async (event, filePath: string, options: any) => {
  return new Promise((resolve, reject) => {
    try {
      // Start Python repair process
      const pythonScript = path.join(__dirname, '../logic/video_repair_engine.py');
      pythonProcess = spawn('python', [pythonScript, 'repair', filePath, JSON.stringify(options)]);
      
      let result = '';
      let error = '';
      
      pythonProcess.stdout?.on('data', (data) => {
        result += data.toString();
        // Send progress updates to renderer
        try {
          const progressData = JSON.parse(data.toString());
          event.sender.send('python:progress', progressData);
        } catch (e) {
          // Not JSON, probably log message
          event.sender.send('python:log', data.toString());
        }
      });
      
      pythonProcess.stderr?.on('data', (data) => {
        error += data.toString();
        event.sender.send('python:error', data.toString());
      });
      
      pythonProcess.on('close', (code) => {
        pythonProcess = null;
        if (code === 0) {
          try {
            resolve(JSON.parse(result));
          } catch (e) {
            resolve({ success: true, message: 'Repair completed' });
          }
        } else {
          reject(new Error(error || `Process exited with code ${code}`));
        }
      });
      
    } catch (error) {
      reject(error);
    }
  });
});

ipcMain.handle('python:start-recovery', async (event, sourcePath: string, outputDir: string, options: any) => {
  return new Promise((resolve, reject) => {
    try {
      const pythonScript = path.join(__dirname, '../logic/video_repair_engine.py');
      pythonProcess = spawn('python', [pythonScript, 'recover', sourcePath, outputDir, JSON.stringify(options)]);
      
      let result = '';
      let error = '';
      
      pythonProcess.stdout?.on('data', (data) => {
        result += data.toString();
        try {
          const progressData = JSON.parse(data.toString());
          event.sender.send('python:progress', progressData);
        } catch (e) {
          event.sender.send('python:log', data.toString());
        }
      });
      
      pythonProcess.stderr?.on('data', (data) => {
        error += data.toString();
        event.sender.send('python:error', data.toString());
      });
      
      pythonProcess.on('close', (code) => {
        pythonProcess = null;
        if (code === 0) {
          try {
            resolve(JSON.parse(result));
          } catch (e) {
            resolve({ success: true, message: 'Recovery completed' });
          }
        } else {
          reject(new Error(error || `Process exited with code ${code}`));
        }
      });
      
    } catch (error) {
      reject(error);
    }
  });
});

ipcMain.handle('python:analyze-file', async (event, filePath: string) => {
  return new Promise((resolve, reject) => {
    try {
      const pythonScript = path.join(__dirname, '../logic/video_repair_engine.py');
      pythonProcess = spawn('python', [pythonScript, 'analyze', filePath]);
      
      let result = '';
      let error = '';
      
      pythonProcess.stdout?.on('data', (data) => {
        result += data.toString();
        event.sender.send('python:log', data.toString());
      });
      
      pythonProcess.stderr?.on('data', (data) => {
        error += data.toString();
        event.sender.send('python:error', data.toString());
      });
      
      pythonProcess.on('close', (code) => {
        pythonProcess = null;
        if (code === 0) {
          try {
            resolve(JSON.parse(result));
          } catch (e) {
            resolve({ success: true, message: 'Analysis completed' });
          }
        } else {
          reject(new Error(error || `Process exited with code ${code}`));
        }
      });
      
    } catch (error) {
      reject(error);
    }
  });
});

// File system operations
ipcMain.handle('fs:select-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    properties: ['openFile'],
    filters: [
      { name: 'Video Files', extensions: ['mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv', 'webm'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('fs:select-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    properties: ['openDirectory']
  });
  
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('fs:save-file', async (event, defaultPath?: string) => {
  const result = await dialog.showSaveDialog(mainWindow!, {
    defaultPath,
    filters: [
      { name: 'Video Files', extensions: ['mp4', 'mov', 'avi'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  
  return result.canceled ? null : result.filePath;
});

// System information
ipcMain.handle('system:get-drives', async () => {
  // This would integrate with the existing drive detection logic
  // For now, return mock data
  return [
    {
      id: 'C:',
      name: 'Local Disk (C:)',
      type: 'hdd',
      totalSpace: 1000000000000,
      freeSpace: 500000000000,
      status: 'healthy',
      isConnected: true
    }
  ];
});

ipcMain.handle('system:get-info', async () => {
  return {
    platform: process.platform,
    arch: process.arch,
    version: app.getVersion(),
    electronVersion: process.versions.electron,
    nodeVersion: process.versions.node
  };
});

// Cancel operations
ipcMain.handle('python:cancel', async () => {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
    return true;
  }
  return false;
});