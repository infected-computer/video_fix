import { Theme } from '@/types';

// Design Kit tokens following 8pt spacing system
export const theme: Theme = {
  spacing: {
    xs: '4px',   // 0.5 * 8pt
    sm: '8px',   // 1 * 8pt
    md: '16px',  // 2 * 8pt
    lg: '24px',  // 3 * 8pt
    xl: '32px',  // 4 * 8pt
    xxl: '48px', // 6 * 8pt
  },
  
  colors: {
    primary: '#007acc',
    secondary: '#6c757d',
    success: '#28a745',
    warning: '#ffc107',
    error: '#dc3545',
    info: '#17a2b8',
    
    background: {
      primary: '#ffffff',
      secondary: '#f8f9fa',
      tertiary: '#e9ecef',
    },
    
    text: {
      primary: '#212529',
      secondary: '#6c757d',
      disabled: '#adb5bd',
    },
    
    border: {
      primary: '#dee2e6',
      secondary: '#e9ecef',
    },
  },
  
  typography: {
    fontFamily: {
      primary: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      monospace: '"SF Mono", Monaco, Inconsolata, "Roboto Mono", "Source Code Pro", monospace',
    },
    
    fontSize: {
      xs: '12px',
      sm: '14px',
      md: '16px',
      lg: '18px',
      xl: '20px',
      xxl: '24px',
    },
    
    fontWeight: {
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    
    lineHeight: {
      tight: 1.25,
      normal: 1.5,
      relaxed: 1.75,
    },
  },
  
  breakpoints: {
    sm: '576px',
    md: '768px',
    lg: '992px',
    xl: '1200px',
  },
  
  shadows: {
    sm: '0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24)',
    md: '0 3px 6px rgba(0, 0, 0, 0.16), 0 3px 6px rgba(0, 0, 0, 0.23)',
    lg: '0 10px 20px rgba(0, 0, 0, 0.19), 0 6px 6px rgba(0, 0, 0, 0.23)',
  },
  
  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '12px',
  },
};

// Dark theme variant
export const darkTheme: Theme = {
  ...theme,
  colors: {
    ...theme.colors,
    background: {
      primary: '#1a1a1a',
      secondary: '#2d2d2d',
      tertiary: '#404040',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b3b3b3',
      disabled: '#666666',
    },
    border: {
      primary: '#404040',
      secondary: '#2d2d2d',
    },
  },
};