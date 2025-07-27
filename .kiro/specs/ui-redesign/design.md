# Design Document

## Overview

This design document outlines the complete UI redesign of PhoenixDRS from a command-line interface to a modern desktop application using the Figma Desktop App Kit design system. The new interface will provide an intuitive, accessible, and professional user experience for video recovery and repair operations while maintaining the powerful functionality of the existing system.

The design follows the Desktop App Kit's component-based architecture, 8pt spacing system, and comprehensive design tokens to ensure consistency, maintainability, and scalability. The application will be built using modern desktop UI frameworks with proper state management and responsive design principles.

## Architecture

### Application Structure
```
PhoenixDRS Desktop Application
├── Main Window (Desktop App Kit Window Component)
│   ├── Sidebar Navigation (Sidebar Component)
│   ├── Main Content Area (Content Container)
│   │   ├── Dashboard View (Card-based Layout)
│   │   ├── Video Repair View (Form + Progress Components)
│   │   ├── Video Recovery View (File Browser + Results)
│   │   ├── File Analysis View (Detail Cards)
│   │   └── Settings View (Form Components)
│   ├── Toolbar (Toolbar Component)
│   └── Status Bar (Status Component)
```

### Technology Stack
- **Frontend Framework**: Electron with React/TypeScript for cross-platform desktop application
- **UI Components**: Custom components based on Desktop App Kit specifications
- **State Management**: Redux Toolkit for application state
- **Styling**: CSS-in-JS with styled-components following Design Kit tokens
- **Backend Integration**: Python backend integration through IPC (Inter-Process Communication)
- **File System**: Native file system APIs with drag-and-drop support

### Design System Integration
- **Component Library**: All components built according to Desktop App Kit specifications
- **Spacing System**: Consistent 8pt grid system throughout the application
- **Typography**: Design Kit typography tokens with proper hierarchy
- **Color System**: Semantic color tokens from the Design Kit palette
- **Icons**: Design Kit icon library with consistent sizing and alignment
- **Interactive States**: Hover, focus, active, and disabled states for all interactive elements

## Components and Interfaces

### 1. Main Window Component
**Purpose**: Root container following Desktop App Kit window specifications
**Design Kit Components Used**: Window, Layout Container
**Key Features**:
- Minimum window size: 1024x768px
- Responsive layout with proper scaling
- Native window controls integration
- Proper focus management

**Layout Structure**:
```
┌─────────────────────────────────────────────────────┐
│ Toolbar (48px height)                              │
├─────────────┬───────────────────────────────────────┤
│ Sidebar     │ Main Content Area                     │
│ (240px)     │                                       │
│             │                                       │
│             │                                       │
│             │                                       │
├─────────────┴───────────────────────────────────────┤
│ Status Bar (32px height)                           │
└─────────────────────────────────────────────────────┘
```

### 2. Sidebar Navigation Component
**Purpose**: Primary navigation using Design Kit Sidebar component
**Design Kit Components Used**: Sidebar, ListItem, Icon, Badge
**Navigation Items**:
- Dashboard (Home icon)
- Video Repair (Tool icon)
- Video Recovery (Search icon)
- File Analysis (Document icon)
- Drive Manager (Storage icon)
- Settings (Gear icon)

**Interactive States**:
- Default: ListItem with icon and label
- Hover: Background color change with smooth transition
- Active: Highlighted background with accent color
- Focus: Keyboard focus ring following Design Kit specs

### 3. Dashboard View Component
**Purpose**: Main overview screen with system status and quick actions
**Design Kit Components Used**: Card, Button, ListItem, Progress, Badge
**Layout Sections**:

#### System Status Cards (Top Row)
- **System Health Card**: CPU, memory, disk usage with progress indicators
- **Active Operations Card**: Current running operations with progress bars
- **Recent Activity Card**: List of recent repair/recovery operations

#### Quick Actions Section (Middle)
- **Primary Action Buttons**: Large buttons for common operations
  - "Repair Video File" (Primary button)
  - "Recover Deleted Videos" (Secondary button)
  - "Analyze File Structure" (Secondary button)

#### Storage Devices Section (Bottom)
- **Connected Drives Card**: List of available storage devices
- Each drive shown as ListItem with:
  - Drive icon and name
  - Capacity and free space
  - Status indicator (healthy/warning/error)

### 4. Video Repair View Component
**Purpose**: Dedicated interface for video file repair operations
**Design Kit Components Used**: Card, Button, Input, Progress, Modal, Alert
**Layout Sections**:

#### File Selection Area
- **Drag & Drop Zone**: Large Card component with dashed border
- **File Browser Button**: Secondary button for traditional file selection
- **Selected File Display**: Card showing file information (name, size, format, path)

#### Repair Options Panel
- **AI Enhancement Toggle**: Switch component with description
- **Output Location**: Input field with folder picker button
- **Advanced Options**: Collapsible section with additional settings

#### Progress & Results Area
- **Progress Indicator**: Linear progress bar with percentage and status text
- **Real-time Log**: Scrollable text area with operation messages
- **Results Summary**: Card with repair statistics and success/error indicators

### 5. Video Recovery View Component
**Purpose**: Interface for recovering deleted video files
**Design Kit Components Used**: Card, Button, Table, Progress, Filter, Search
**Layout Sections**:

#### Source Selection
- **Drive Selection**: Dropdown with available drives
- **Scan Options**: Checkboxes for file types and scan depth
- **Scan Button**: Primary button to start recovery scan

#### Scan Results
- **Results Table**: Data table with columns:
  - Thumbnail (if available)
  - File name
  - Size
  - Format
  - Recovery confidence
  - Actions (Preview, Recover)
- **Filter Controls**: Search input and filter dropdowns
- **Bulk Actions**: Select all, recover selected buttons

#### Recovery Progress
- **Progress Panel**: Shows current recovery operation status
- **Destination Folder**: Input with folder picker
- **Recovery Log**: Real-time status messages

### 6. File Analysis View Component
**Purpose**: Detailed analysis of video file structure and metadata
**Design Kit Components Used**: Card, Table, Tree, Tabs, Code Block
**Layout Sections**:

#### File Information Panel
- **Basic Info Card**: File name, size, format, creation date
- **Technical Details Card**: Codec, resolution, bitrate, duration
- **Health Status Card**: Corruption indicators and repair suggestions

#### Structure Analysis
- **File Structure Tree**: Hierarchical view of file containers and streams
- **Hex Viewer**: Raw file data display with highlighting
- **Metadata Table**: Key-value pairs of file metadata

#### Analysis Results
- **Issues Found**: List of detected problems with severity indicators
- **Repair Recommendations**: Suggested actions with confidence levels
- **Export Options**: Buttons to save analysis report

### 7. Settings View Component
**Purpose**: Application configuration and preferences
**Design Kit Components Used**: Card, Input, Switch, Select, Button, Tabs
**Settings Categories**:

#### General Settings
- **Language Selection**: Dropdown with RTL support indicator
- **Theme Selection**: Light/Dark mode toggle
- **Auto-save Options**: Switches for various auto-save features

#### Performance Settings
- **CPU Usage Limit**: Slider component
- **Memory Allocation**: Input with validation
- **Parallel Processing**: Switch with worker count input

#### Advanced Settings
- **Logging Level**: Select dropdown
- **Temporary Files Location**: Input with folder picker
- **Backend Preference**: Radio buttons (Python/C++/Auto)

## Data Models

### Application State Structure
```typescript
interface ApplicationState {
  ui: {
    currentView: 'dashboard' | 'repair' | 'recovery' | 'analysis' | 'settings';
    sidebarCollapsed: boolean;
    theme: 'light' | 'dark';
    language: 'en' | 'he';
  };
  system: {
    drives: DriveInfo[];
    systemHealth: SystemHealth;
    activeOperations: Operation[];
  };
  repair: {
    selectedFile: FileInfo | null;
    options: RepairOptions;
    progress: ProgressInfo | null;
    results: RepairResult | null;
  };
  recovery: {
    selectedDrive: string | null;
    scanOptions: ScanOptions;
    scanResults: RecoveredFile[];
    recoveryProgress: ProgressInfo | null;
  };
  analysis: {
    selectedFile: FileInfo | null;
    analysisResults: AnalysisResult | null;
  };
  settings: UserSettings;
}
```

### Core Data Types
```typescript
interface FileInfo {
  path: string;
  name: string;
  size: number;
  format: string;
  lastModified: Date;
  isCorrupted: boolean;
}

interface DriveInfo {
  id: string;
  name: string;
  type: 'hdd' | 'ssd' | 'usb' | 'optical';
  totalSpace: number;
  freeSpace: number;
  status: 'healthy' | 'warning' | 'error';
  isConnected: boolean;
}

interface ProgressInfo {
  percentage: number;
  status: string;
  estimatedTimeRemaining: number;
  currentOperation: string;
}

interface RepairResult {
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
```

## Error Handling

### Error State Management
Following the Design Kit's error handling patterns:

#### Error Types and Display
1. **Validation Errors**: Inline field validation with error text below inputs
2. **Operation Errors**: Alert components with error icon and descriptive message
3. **System Errors**: Modal dialogs for critical errors requiring user attention
4. **Network Errors**: Toast notifications for connectivity issues

#### Error Recovery Patterns
- **Retry Mechanisms**: Buttons to retry failed operations
- **Fallback Options**: Alternative actions when primary operations fail
- **Error Reporting**: Option to send error reports to support
- **Graceful Degradation**: Reduced functionality when components fail

#### Error Message Design
- **Clear Language**: User-friendly error messages avoiding technical jargon
- **Actionable Guidance**: Specific steps users can take to resolve issues
- **Context Awareness**: Error messages relevant to current operation
- **Accessibility**: Proper ARIA labels and screen reader support

## Testing Strategy

### Component Testing
- **Unit Tests**: Individual component functionality and props handling
- **Integration Tests**: Component interaction and data flow
- **Visual Regression Tests**: Design consistency and responsive behavior
- **Accessibility Tests**: Keyboard navigation, screen reader compatibility, color contrast

### User Experience Testing
- **Usability Testing**: Task completion rates and user satisfaction
- **Performance Testing**: Application responsiveness and resource usage
- **Cross-platform Testing**: Windows, macOS, and Linux compatibility
- **Internationalization Testing**: RTL layout and Hebrew text rendering

### Backend Integration Testing
- **API Integration**: Python backend communication through IPC
- **File System Operations**: Drag-and-drop, file selection, and path handling
- **Error Handling**: Backend error propagation and user notification
- **Performance Testing**: Large file handling and long-running operations

### Design System Compliance Testing
- **Component Consistency**: Adherence to Design Kit specifications
- **Spacing Validation**: 8pt grid system compliance
- **Typography Testing**: Font rendering and hierarchy consistency
- **Interactive State Testing**: Hover, focus, active, and disabled states
- **Color Contrast Testing**: Accessibility compliance for all color combinations

## Implementation Phases

### Phase 1: Foundation Setup
- Set up Electron + React + TypeScript project structure
- Implement Design Kit component library
- Create basic window layout with Sidebar and main content area
- Establish state management with Redux Toolkit
- Set up styling system with Design Kit tokens

### Phase 2: Core Views Implementation
- Implement Dashboard view with system status cards
- Create Video Repair view with file selection and progress tracking
- Build Video Recovery view with scan results table
- Develop File Analysis view with detailed information display

### Phase 3: Advanced Features
- Add drag-and-drop functionality throughout the application
- Implement real-time progress tracking and status updates
- Create comprehensive error handling and recovery mechanisms
- Add settings view with all configuration options

### Phase 4: Polish and Accessibility
- Implement RTL support for Hebrew language
- Add comprehensive keyboard navigation
- Ensure full screen reader compatibility
- Optimize performance for large file operations
- Add comprehensive testing suite

### Phase 5: Integration and Deployment
- Complete Python backend integration through IPC
- Implement auto-updater functionality
- Create installer packages for all platforms
- Add telemetry and crash reporting
- Final testing and quality assurance