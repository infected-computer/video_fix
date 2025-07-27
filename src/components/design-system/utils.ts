import { designTokens, darkTokens } from '@/styles/tokens';

// Utility functions for Design System components

/**
 * Get spacing value from design tokens
 */
export const getSpacing = (value: keyof typeof designTokens.spacing): string => {
  return designTokens.spacing[value];
};

/**
 * Get color value from design tokens
 */
export const getColor = (color: string, shade?: number, isDark = false): string => {
  const tokens = isDark ? darkTokens : designTokens;
  
  if (shade && tokens.colors[color as keyof typeof tokens.colors]) {
    const colorGroup = tokens.colors[color as keyof typeof tokens.colors] as any;
    return colorGroup[shade] || colorGroup[500] || color;
  }
  
  // Handle semantic colors
  if (color.includes('.')) {
    const [group, variant] = color.split('.');
    const colorGroup = tokens.colors[group as keyof typeof tokens.colors] as any;
    if (typeof colorGroup === 'object' && colorGroup !== null) {
      return colorGroup[variant] || color;
    }
  }
  
  const colorValue = tokens.colors[color as keyof typeof tokens.colors];
  if (typeof colorValue === 'string') {
    return colorValue;
  }
  
  return color;
};

/**
 * Get typography value from design tokens
 */
export const getTypography = (property: string, value: string): string => {
  const typography = designTokens.typography as any;
  return typography[property]?.[value] || value;
};

/**
 * Get shadow value from design tokens
 */
export const getShadow = (value: keyof typeof designTokens.shadows): string => {
  return designTokens.shadows[value];
};

/**
 * Get border radius value from design tokens
 */
export const getBorderRadius = (value: keyof typeof designTokens.borderRadius): string => {
  return designTokens.borderRadius[value];
};

/**
 * Generate focus ring styles
 */
export const getFocusRing = (color = 'primary.500'): string => {
  return `
    outline: 2px solid ${getColor(color)};
    outline-offset: 2px;
  `;
};

/**
 * Generate hover styles for interactive elements
 */
export const getHoverStyles = (baseColor: string, isDark = false): string => {
  const hoverColor = isDark ? 
    getColor(baseColor.replace('500', '400'), undefined, isDark) :
    getColor(baseColor.replace('500', '600'), undefined, isDark);
  
  return `
    background-color: ${hoverColor};
    transform: translateY(-1px);
    box-shadow: ${getShadow('md')};
  `;
};

/**
 * Generate active styles for interactive elements
 */
export const getActiveStyles = (baseColor: string, isDark = false): string => {
  const activeColor = isDark ?
    getColor(baseColor.replace('500', '300'), undefined, isDark) :
    getColor(baseColor.replace('500', '700'), undefined, isDark);
  
  return `
    background-color: ${activeColor};
    transform: translateY(0);
    box-shadow: ${getShadow('sm')};
  `;
};

/**
 * Generate disabled styles for interactive elements
 */
export const getDisabledStyles = (isDark = false): string => {
  return `
    opacity: 0.6;
    cursor: not-allowed;
    pointer-events: none;
    background-color: ${getColor('neutral.200', undefined, isDark)};
    color: ${getColor('neutral.400', undefined, isDark)};
  `;
};

/**
 * Generate responsive styles based on breakpoints
 */
export const getResponsiveStyles = (styles: Record<string, string>): string => {
  return Object.entries(styles)
    .map(([breakpoint, style]) => {
      if (breakpoint === 'base') return style;
      const bp = designTokens.breakpoints[breakpoint as keyof typeof designTokens.breakpoints];
      return `@media (min-width: ${bp}) { ${style} }`;
    })
    .join('\n');
};

/**
 * Generate animation styles
 */
export const getAnimation = (
  property: string,
  duration: '75' | '100' | '150' | '200' | '300' | '500' | '700' | '1000' = '200',
  timing: keyof typeof designTokens.timing = 'inOut'
): string => {
  return `
    transition: ${property} ${designTokens.duration[duration]} ${designTokens.timing[timing]};
  `;
};

/**
 * Generate RTL-aware styles
 */
export const getRTLStyles = (ltrStyles: string, rtlStyles: string): string => {
  return `
    ${ltrStyles}
    
    [dir="rtl"] & {
      ${rtlStyles}
    }
  `;
};

/**
 * Generate high contrast mode styles
 */
export const getHighContrastStyles = (styles: string): string => {
  return `
    @media (prefers-contrast: high) {
      ${styles}
    }
  `;
};

/**
 * Generate reduced motion styles
 */
export const getReducedMotionStyles = (styles: string): string => {
  return `
    @media (prefers-reduced-motion: reduce) {
      ${styles}
    }
  `;
};

/**
 * Clamp a value between min and max
 */
export const clamp = (value: number, min: number, max: number): number => {
  return Math.min(Math.max(value, min), max);
};

/**
 * Convert rem to px
 */
export const remToPx = (rem: number, baseFontSize = 16): number => {
  return rem * baseFontSize;
};

/**
 * Convert px to rem
 */
export const pxToRem = (px: number, baseFontSize = 16): number => {
  return px / baseFontSize;
};

/**
 * Generate component size styles
 */
export const getSizeStyles = (size: 'sm' | 'md' | 'lg', component: 'button' | 'input' | 'card'): string => {
  const sizeMap = {
    button: {
      sm: `
        height: 32px;
        padding: 0 ${getSpacing(3)};
        font-size: ${getTypography('fontSize', 'sm')};
      `,
      md: `
        height: 40px;
        padding: 0 ${getSpacing(4)};
        font-size: ${getTypography('fontSize', 'base')};
      `,
      lg: `
        height: 48px;
        padding: 0 ${getSpacing(6)};
        font-size: ${getTypography('fontSize', 'lg')};
      `,
    },
    input: {
      sm: `
        height: 32px;
        padding: 0 ${getSpacing(3)};
        font-size: ${getTypography('fontSize', 'sm')};
      `,
      md: `
        height: 40px;
        padding: 0 ${getSpacing(4)};
        font-size: ${getTypography('fontSize', 'base')};
      `,
      lg: `
        height: 48px;
        padding: 0 ${getSpacing(4)};
        font-size: ${getTypography('fontSize', 'lg')};
      `,
    },
    card: {
      sm: `padding: ${getSpacing(3)};`,
      md: `padding: ${getSpacing(4)};`,
      lg: `padding: ${getSpacing(6)};`,
    },
  };
  
  return sizeMap[component][size];
};

/**
 * Generate z-index value
 */
export const getZIndex = (level: keyof typeof designTokens.zIndex): string | number => {
  return designTokens.zIndex[level];
};