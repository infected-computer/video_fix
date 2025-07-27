import { ReactNode, HTMLAttributes, ButtonHTMLAttributes, InputHTMLAttributes } from 'react';

// Base component props
export interface BaseProps {
  className?: string;
  children?: ReactNode;
  'data-testid'?: string;
}

// Size variants
export type Size = 'xs' | 'sm' | 'md' | 'lg' | 'xl';

// Color variants
export type ColorVariant = 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info' | 'neutral';

// Interactive states
export interface InteractiveStates {
  hover?: boolean;
  focus?: boolean;
  active?: boolean;
  disabled?: boolean;
  loading?: boolean;
}

// Button variants and props
export type ButtonVariant = 'primary' | 'secondary' | 'tertiary' | 'ghost' | 'link';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends BaseProps, ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  color?: ColorVariant;
  fullWidth?: boolean;
  loading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  'aria-label'?: string;
}

// Card variants and props
export type CardVariant = 'elevated' | 'outlined' | 'filled';
export type CardPadding = 'none' | 'sm' | 'md' | 'lg';

export interface CardProps extends BaseProps, HTMLAttributes<HTMLDivElement> {
  variant?: CardVariant;
  padding?: CardPadding;
  hoverable?: boolean;
  clickable?: boolean;
  selected?: boolean;
}

// Input variants and props
export type InputVariant = 'outlined' | 'filled' | 'underlined';
export type InputSize = 'sm' | 'md' | 'lg';

export interface InputProps extends BaseProps, Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  variant?: InputVariant;
  size?: InputSize;
  label?: string;
  helperText?: string;
  error?: boolean;
  errorMessage?: string;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  fullWidth?: boolean;
}

// Select props
export interface SelectOption {
  value: string | number;
  label: string;
  disabled?: boolean;
}

export interface SelectProps extends BaseProps {
  options: SelectOption[];
  value?: string | number;
  defaultValue?: string | number;
  placeholder?: string;
  label?: string;
  helperText?: string;
  error?: boolean;
  errorMessage?: string;
  disabled?: boolean;
  fullWidth?: boolean;
  size?: InputSize;
  onChange?: (value: string | number) => void;
}

// Switch props
export interface SwitchProps extends BaseProps {
  checked?: boolean;
  defaultChecked?: boolean;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  description?: string;
  onChange?: (checked: boolean) => void;
}

// Slider props
export interface SliderProps extends BaseProps {
  value?: number;
  defaultValue?: number;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  label?: string;
  showValue?: boolean;
  formatValue?: (value: number) => string;
  onChange?: (value: number) => void;
}

// Checkbox props
export interface CheckboxProps extends BaseProps {
  checked?: boolean;
  defaultChecked?: boolean;
  indeterminate?: boolean;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  description?: string;
  error?: boolean;
  onChange?: (checked: boolean) => void;
}

// Radio props
export interface RadioProps extends BaseProps {
  checked?: boolean;
  value: string | number;
  name?: string;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  description?: string;
  onChange?: (value: string | number) => void;
}

// ListItem props
export interface ListItemProps extends BaseProps, HTMLAttributes<HTMLDivElement> {
  primary: string;
  secondary?: string;
  tertiary?: string;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  avatar?: ReactNode;
  badge?: ReactNode;
  active?: boolean;
  disabled?: boolean;
  divider?: boolean;
  clickable?: boolean;
}

// Progress props
export type ProgressVariant = 'linear' | 'circular';
export type ProgressSize = 'sm' | 'md' | 'lg';

export interface ProgressProps extends BaseProps {
  variant?: ProgressVariant;
  value?: number;
  max?: number;
  size?: ProgressSize;
  color?: ColorVariant;
  indeterminate?: boolean;
  label?: string;
  showValue?: boolean;
  formatValue?: (value: number, max: number) => string;
}

// Alert props
export type AlertVariant = 'filled' | 'outlined' | 'standard';
export type AlertSeverity = 'success' | 'info' | 'warning' | 'error';

export interface AlertProps extends BaseProps {
  variant?: AlertVariant;
  severity?: AlertSeverity;
  title?: string;
  closable?: boolean;
  icon?: ReactNode;
  action?: ReactNode;
  onClose?: () => void;
}

// Modal props
export interface ModalProps extends BaseProps {
  open: boolean;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  closable?: boolean;
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
  footer?: ReactNode;
  onClose?: () => void;
}

// Badge props
export type BadgeVariant = 'filled' | 'outlined' | 'dot';
export type BadgeSize = 'sm' | 'md' | 'lg';

export interface BadgeProps extends BaseProps {
  variant?: BadgeVariant;
  size?: BadgeSize;
  color?: ColorVariant;
  content?: string | number;
  max?: number;
  showZero?: boolean;
  invisible?: boolean;
}

// Icon props
export interface IconProps extends BaseProps {
  name: string;
  size?: Size;
  color?: string;
  'aria-label'?: string;
}

// Spinner props
export interface SpinnerProps extends BaseProps {
  size?: Size;
  color?: ColorVariant;
  thickness?: number;
}

// Skeleton props
export interface SkeletonProps extends BaseProps {
  variant?: 'text' | 'rectangular' | 'circular';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | false;
}

// Divider props
export interface DividerProps extends BaseProps {
  orientation?: 'horizontal' | 'vertical';
  variant?: 'fullWidth' | 'inset' | 'middle';
  textAlign?: 'left' | 'center' | 'right';
}

// Table props
export interface TableColumn {
  key: string;
  title: string;
  width?: string | number;
  sortable?: boolean;
  render?: (value: any, record: any, index: number) => ReactNode;
}

export interface TableProps extends BaseProps {
  columns: TableColumn[];
  data: any[];
  loading?: boolean;
  empty?: ReactNode;
  pagination?: {
    current: number;
    pageSize: number;
    total: number;
    onChange: (page: number, pageSize: number) => void;
  };
  selection?: {
    selectedRowKeys: string[];
    onChange: (selectedRowKeys: string[], selectedRows: any[]) => void;
  };
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  onSort?: (sortBy: string, sortOrder: 'asc' | 'desc') => void;
}

// Tabs props
export interface TabItem {
  key: string;
  label: string;
  content: ReactNode;
  disabled?: boolean;
  icon?: ReactNode;
}

export interface TabsProps extends BaseProps {
  items: TabItem[];
  activeKey?: string;
  defaultActiveKey?: string;
  variant?: 'line' | 'card' | 'pill';
  size?: 'sm' | 'md' | 'lg';
  onChange?: (activeKey: string) => void;
}

// Sidebar props
export interface SidebarItem {
  key: string;
  label: string;
  icon?: ReactNode;
  badge?: ReactNode;
  children?: SidebarItem[];
  disabled?: boolean;
}

export interface SidebarProps extends BaseProps {
  items: SidebarItem[];
  activeKey?: string;
  collapsed?: boolean;
  collapsible?: boolean;
  width?: number;
  onItemClick?: (key: string, item: SidebarItem) => void;
  onCollapse?: (collapsed: boolean) => void;
}

// Toolbar props
export interface ToolbarProps extends BaseProps {
  title?: string;
  subtitle?: string;
  leftContent?: ReactNode;
  rightContent?: ReactNode;
  height?: number;
}

// Toast props
export type ToastType = 'success' | 'error' | 'warning' | 'info';
export type ToastPosition = 'top-left' | 'top-center' | 'top-right' | 'bottom-left' | 'bottom-center' | 'bottom-right';

export interface ToastProps extends BaseProps {
  type?: ToastType;
  title?: string;
  description?: string;
  duration?: number;
  closable?: boolean;
  action?: ReactNode;
  onClose?: () => void;
}

// Tooltip props
export type TooltipPlacement = 'top' | 'bottom' | 'left' | 'right' | 'top-start' | 'top-end' | 'bottom-start' | 'bottom-end' | 'left-start' | 'left-end' | 'right-start' | 'right-end';

export interface TooltipProps extends BaseProps {
  content: ReactNode;
  placement?: TooltipPlacement;
  trigger?: 'hover' | 'click' | 'focus';
  disabled?: boolean;
  delay?: number;
  arrow?: boolean;
}

// Avatar props
export interface AvatarProps extends BaseProps {
  src?: string;
  alt?: string;
  size?: Size;
  shape?: 'circle' | 'square';
  fallback?: string;
  color?: ColorVariant;
}