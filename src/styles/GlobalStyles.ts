import { createGlobalStyle } from 'styled-components';

export const GlobalStyles = createGlobalStyle`
  /* Reset and base styles */
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  html {
    font-size: 16px;
    line-height: 1.5;
  }

  body {
    font-family: ${({ theme }) => theme.typography.fontFamily.primary};
    font-size: ${({ theme }) => theme.typography.fontSize.md};
    font-weight: ${({ theme }) => theme.typography.fontWeight.normal};
    line-height: ${({ theme }) => theme.typography.lineHeight.normal};
    color: ${({ theme }) => theme.colors.text.primary};
    background-color: ${({ theme }) => theme.colors.background.primary};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    overflow: hidden; /* Prevent body scroll in desktop app */
  }

  /* Typography hierarchy */
  h1 {
    font-size: ${({ theme }) => theme.typography.fontSize.xxl};
    font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
    line-height: ${({ theme }) => theme.typography.lineHeight.tight};
    margin-bottom: ${({ theme }) => theme.spacing.md};
  }

  h2 {
    font-size: ${({ theme }) => theme.typography.fontSize.xl};
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
    line-height: ${({ theme }) => theme.typography.lineHeight.tight};
    margin-bottom: ${({ theme }) => theme.spacing.sm};
  }

  h3 {
    font-size: ${({ theme }) => theme.typography.fontSize.lg};
    font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
    line-height: ${({ theme }) => theme.typography.lineHeight.normal};
    margin-bottom: ${({ theme }) => theme.spacing.sm};
  }

  h4 {
    font-size: ${({ theme }) => theme.typography.fontSize.md};
    font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
    line-height: ${({ theme }) => theme.typography.lineHeight.normal};
    margin-bottom: ${({ theme }) => theme.spacing.xs};
  }

  p {
    margin-bottom: ${({ theme }) => theme.spacing.sm};
  }

  /* Interactive elements */
  button {
    font-family: inherit;
    cursor: pointer;
    border: none;
    background: none;
    outline: none;
    
    &:focus-visible {
      outline: 2px solid ${({ theme }) => theme.colors.primary};
      outline-offset: 2px;
    }
  }

  input, textarea, select {
    font-family: inherit;
    font-size: inherit;
    border: 1px solid ${({ theme }) => theme.colors.border.primary};
    border-radius: ${({ theme }) => theme.borderRadius.sm};
    padding: ${({ theme }) => theme.spacing.sm};
    
    &:focus {
      outline: none;
      border-color: ${({ theme }) => theme.colors.primary};
      box-shadow: 0 0 0 2px ${({ theme }) => theme.colors.primary}20;
    }
    
    &:disabled {
      background-color: ${({ theme }) => theme.colors.background.tertiary};
      color: ${({ theme }) => theme.colors.text.disabled};
      cursor: not-allowed;
    }
  }

  /* Scrollbars */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: ${({ theme }) => theme.colors.background.secondary};
  }

  ::-webkit-scrollbar-thumb {
    background: ${({ theme }) => theme.colors.border.primary};
    border-radius: ${({ theme }) => theme.borderRadius.sm};
    
    &:hover {
      background: ${({ theme }) => theme.colors.text.secondary};
    }
  }

  /* Selection */
  ::selection {
    background-color: ${({ theme }) => theme.colors.primary}30;
    color: ${({ theme }) => theme.colors.text.primary};
  }

  /* Focus management */
  .focus-visible {
    outline: 2px solid ${({ theme }) => theme.colors.primary};
    outline-offset: 2px;
  }

  /* Accessibility */
  @media (prefers-reduced-motion: reduce) {
    * {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }

  /* High contrast mode support */
  @media (prefers-contrast: high) {
    button, input, select, textarea {
      border-width: 2px;
    }
  }

  /* RTL support */
  [dir="rtl"] {
    text-align: right;
    
    /* Flip margins and paddings */
    .margin-left { margin-right: inherit; margin-left: 0; }
    .margin-right { margin-left: inherit; margin-right: 0; }
    .padding-left { padding-right: inherit; padding-left: 0; }
    .padding-right { padding-left: inherit; padding-right: 0; }
    
    /* Flip transforms */
    .transform-flip {
      transform: scaleX(-1);
    }
  }

  /* Loading states */
  .loading {
    pointer-events: none;
    opacity: 0.6;
  }

  /* Error states */
  .error {
    color: ${({ theme }) => theme.colors.error};
    border-color: ${({ theme }) => theme.colors.error};
  }

  /* Success states */
  .success {
    color: ${({ theme }) => theme.colors.success};
    border-color: ${({ theme }) => theme.colors.success};
  }

  /* Warning states */
  .warning {
    color: ${({ theme }) => theme.colors.warning};
    border-color: ${({ theme }) => theme.colors.warning};
  }

  /* Utility classes */
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  .truncate {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .text-center { text-align: center; }
  .text-left { text-align: left; }
  .text-right { text-align: right; }

  /* Spacing utilities following 8pt system */
  .m-0 { margin: 0; }
  .m-1 { margin: ${({ theme }) => theme.spacing.xs}; }
  .m-2 { margin: ${({ theme }) => theme.spacing.sm}; }
  .m-3 { margin: ${({ theme }) => theme.spacing.md}; }
  .m-4 { margin: ${({ theme }) => theme.spacing.lg}; }
  .m-5 { margin: ${({ theme }) => theme.spacing.xl}; }
  .m-6 { margin: ${({ theme }) => theme.spacing.xxl}; }

  .p-0 { padding: 0; }
  .p-1 { padding: ${({ theme }) => theme.spacing.xs}; }
  .p-2 { padding: ${({ theme }) => theme.spacing.sm}; }
  .p-3 { padding: ${({ theme }) => theme.spacing.md}; }
  .p-4 { padding: ${({ theme }) => theme.spacing.lg}; }
  .p-5 { padding: ${({ theme }) => theme.spacing.xl}; }
  .p-6 { padding: ${({ theme }) => theme.spacing.xxl}; }
`;