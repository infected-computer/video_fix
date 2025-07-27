// Core application types
export interface FileInfo {
  path: string;
  name: string;
  size: number;
  format: string;
  lastModified: Date;
  isCorrupted: boolean;
}

export interface DriveInfo {
  id: string;
  name: string;
  type: 'hdd' | 'ssd' | 'usb' | 'optical';
  totalSpace: number;
  freeSpace: number;
  status: 'healthy' | 'warning' | 'error';
  isConnected: boolean;
}

export interface ProgressInfo {
  percentage: number;
  status: string;
  estimatedTimeRemaining: number;
  currentOperation: string;
}

export interface RepairResult {
  success: boolean;
  originalFile: string;
  repairedFile: string;
  errorsFound: string[];
  errorsFixed: string[];
  repairTime: number;
  fileSizeBefore: number;
  fileSizeAfter: number;
  aiEnhanced: boolean;
}

export interface RecoveredFile {
  id: string;
  name: string;
  size: number;
  format: string;
  confidence: number;
  thumbnail?: string;
  path: string;
}

export interface AnalysisResult {
  fileInfo: FileInfo;
  technicalDetails: {
    codec: string;
    resolution: string;
    bitrate: number;
    duration: number;
    frameRate: number;
  };
  healthStatus: {
    isCorrupted: boolean;
    issues: string[];
    severity: 'low' | 'medium' | 'high';
  };
  structure: any; // File structure tree
  metadata: Record<string, any>;
}

export interface SystemHealth {
  cpu: number;
  memory: number;
  disk: number;
  status: 'good' | 'warning' | 'critical';
}

export interface Operation {
  id: string;
  type: 'repair' | 'recovery' | 'analysis';
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  progress: ProgressInfo;
  startTime: Date;
  endTime?: Date;
}

export interface UserSettings {
  language: 'en' | 'he';
  theme: 'light' | 'dark';
  autoSave: boolean;
  cpuLimit: number;
  memoryLimit: number;
  parallelProcessing: boolean;
  workerCount: number;
  loggingLevel: 'debug' | 'info' | 'warn' | 'error';
  tempDirectory: string;
  backendPreference: 'python' | 'cpp' | 'auto';
}

// UI State types
export type ViewType = 'dashboard' | 'repair' | 'recovery' | 'analysis' | 'settings';

export interface UIState {
  currentView: ViewType;
  sidebarCollapsed: boolean;
  theme: 'light' | 'dark';
  language: 'en' | 'he';
  isLoading: boolean;
  error: string | null;
}

// Component prop types
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface ButtonProps extends BaseComponentProps {
  variant?: 'primary' | 'secondary' | 'tertiary';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
  'aria-label'?: string;
}

export interface CardProps extends BaseComponentProps {
  variant?: 'elevated' | 'outlined' | 'flat';
  padding?: 'none' | 'small' | 'medium' | 'large';
  hoverable?: boolean;
  onClick?: () => void;
}

export interface ListItemProps extends BaseComponentProps {
  icon?: React.ReactNode;
  primary: string;
  secondary?: string;
  badge?: string | number;
  active?: boolean;
  disabled?: boolean;
  onClick?: () => void;
}

// Theme types
export interface SpacingTokens {
  xs: string; // 4px
  sm: string; // 8px
  md: string; // 16px
  lg: string; // 24px
  xl: string; // 32px
  xxl: string; // 48px
}

export interface ColorTokens {
  primary: string;
  secondary: string;
  success: string;
  warning: string;
  error: string;
  info: string;
  background: {
    primary: string;
    secondary: string;
    tertiary: string;
  };
  text: {
    primary: string;
    secondary: string;
    disabled: string;
  };
  border: {
    primary: string;
    secondary: string;
  };
}

export interface TypographyTokens {
  fontFamily: {
    primary: string;
    monospace: string;
  };
  fontSize: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
    xxl: string;
  };
  fontWeight: {
    normal: number;
    medium: number;
    semibold: number;
    bold: number;
  };
  lineHeight: {
    tight: number;
    normal: number;
    relaxed: number;
  };
}

export interface Theme {
  spacing: SpacingTokens;
  colors: ColorTokens;
  typography: TypographyTokens;
  breakpoints: {
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  shadows: {
    sm: string;
    md: string;
    lg: string;
  };
  borderRadius: {
    sm: string;
    md: string;
    lg: string;
  };
}