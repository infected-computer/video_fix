import '@testing-library/jest-dom';

// Mock Electron API for tests
const mockElectronAPI = {
  python: {
    startRepair: jest.fn(),
    startRecovery: jest.fn(),
    analyzeFile: jest.fn(),
    cancel: jest.fn(),
    onProgress: jest.fn(),
    onLog: jest.fn(),
    onError: jest.fn(),
    removeAllListeners: jest.fn(),
  },
  fs: {
    selectFile: jest.fn(),
    selectFolder: jest.fn(),
    saveFile: jest.fn(),
  },
  system: {
    getDrives: jest.fn(),
    getInfo: jest.fn(),
  },
};

Object.defineProperty(window, 'electronAPI', {
  value: mockElectronAPI,
  writable: true,
});

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock IntersectionObserver
global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Suppress console warnings in tests
const originalError = console.error;
beforeAll(() => {
  console.error = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('Warning: ReactDOM.render is no longer supported')
    ) {
      return;
    }
    originalError.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
});