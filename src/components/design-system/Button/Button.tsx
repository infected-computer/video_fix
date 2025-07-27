import React, { forwardRef } from 'react';
import styled, { css } from 'styled-components';
import { ButtonProps } from '../types';
import { 
  getSpacing, 
  getColor, 
  getTypography, 
  getBorderRadius, 
  getShadow,
  getAnimation,
  getFocusRing,
  getDisabledStyles,
  getSizeStyles
} from '../utils';
import Spinner from '../Spinner';

const ButtonBase = styled.button<{
  $variant: ButtonProps['variant'];
  $size: ButtonProps['size'];
  $color: ButtonProps['color'];
  $fullWidth: boolean;
  $loading: boolean;
}>`
  /* Base styles */
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: ${getSpacing(2)};
  border: none;
  border-radius: ${getBorderRadius('base')};
  font-family: ${getTypography('fontFamily', 'primary')};
  font-weight: ${getTypography('fontWeight', 'medium')};
  text-decoration: none;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  position: relative;
  overflow: hidden;
  
  /* Animation */
  ${getAnimation('all', '150')}
  
  /* Size styles */
  ${({ $size }) => getSizeStyles($size || 'md', 'button')}
  
  /* Full width */
  ${({ $fullWidth }) => $fullWidth && css`
    width: 100%;
  `}
  
  /* Loading state */
  ${({ $loading }) => $loading && css`
    pointer-events: none;
    
    > *:not(.spinner) {
      opacity: 0;
    }
  `}
  
  /* Variant styles */
  ${({ $variant, $color }) => {
    const baseColor = $color || 'primary';
    
    switch ($variant) {
      case 'primary':
        return css`
          background-color: ${getColor(`${baseColor}.500`)};
          color: ${getColor('neutral.0')};
          box-shadow: ${getShadow('sm')};
          
          &:hover:not(:disabled) {
            background-color: ${getColor(`${baseColor}.600`)};
            box-shadow: ${getShadow('md')};
            transform: translateY(-1px);
          }
          
          &:active:not(:disabled) {
            background-color: ${getColor(`${baseColor}.700`)};
            box-shadow: ${getShadow('sm')};
            transform: translateY(0);
          }
        `;
        
      case 'secondary':
        return css`
          background-color: ${getColor('neutral.0')};
          color: ${getColor(`${baseColor}.500`)};
          border: 1px solid ${getColor(`${baseColor}.300`)};
          
          &:hover:not(:disabled) {
            background-color: ${getColor(`${baseColor}.50`)};
            border-color: ${getColor(`${baseColor}.400`)};
            transform: translateY(-1px);
          }
          
          &:active:not(:disabled) {
            background-color: ${getColor(`${baseColor}.100`)};
            border-color: ${getColor(`${baseColor}.500`)};
            transform: translateY(0);
          }
        `;
        
      case 'tertiary':
        return css`
          background-color: transparent;
          color: ${getColor(`${baseColor}.500`)};
          
          &:hover:not(:disabled) {
            background-color: ${getColor(`${baseColor}.50`)};
            transform: translateY(-1px);
          }
          
          &:active:not(:disabled) {
            background-color: ${getColor(`${baseColor}.100`)};
            transform: translateY(0);
          }
        `;
        
      case 'ghost':
        return css`
          background-color: transparent;
          color: ${getColor('text.primary')};
          
          &:hover:not(:disabled) {
            background-color: ${getColor('neutral.100')};
          }
          
          &:active:not(:disabled) {
            background-color: ${getColor('neutral.200')};
          }
        `;
        
      case 'link':
        return css`
          background-color: transparent;
          color: ${getColor(`${baseColor}.500`)};
          text-decoration: underline;
          box-shadow: none;
          
          &:hover:not(:disabled) {
            color: ${getColor(`${baseColor}.600`)};
            text-decoration: none;
          }
          
          &:active:not(:disabled) {
            color: ${getColor(`${baseColor}.700`)};
          }
        `;
        
      default:
        return css`
          background-color: ${getColor(`${baseColor}.500`)};
          color: ${getColor('neutral.0')};
        `;
    }
  }}
  
  /* Focus styles */
  &:focus-visible {
    ${({ $color }) => getFocusRing(getColor(`${$color || 'primary'}.500`))}
  }
  
  /* Disabled styles */
  &:disabled {
    ${getDisabledStyles()}
  }
  
  /* RTL support */
  [dir="rtl"] & {
    /* Icons will be flipped automatically by the Icon component */
  }
  
  /* High contrast mode */
  @media (prefers-contrast: high) {
    border-width: 2px;
  }
  
  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    transition: none;
    transform: none !important;
  }
`;

const SpinnerWrapper = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
`;

const IconWrapper = styled.span<{ $position: 'left' | 'right' }>`
  display: inline-flex;
  align-items: center;
  
  ${({ $position }) => $position === 'left' && css`
    margin-right: ${getSpacing(1)};
  `}
  
  ${({ $position }) => $position === 'right' && css`
    margin-left: ${getSpacing(1)};
  `}
  
  [dir="rtl"] & {
    ${({ $position }) => $position === 'left' && css`
      margin-right: 0;
      margin-left: ${getSpacing(1)};
    `}
    
    ${({ $position }) => $position === 'right' && css`
      margin-left: 0;
      margin-right: ${getSpacing(1)};
    `}
  }
`;

const Button = forwardRef<HTMLButtonElement, ButtonProps>(({
  variant = 'primary',
  size = 'md',
  color = 'primary',
  fullWidth = false,
  loading = false,
  disabled = false,
  leftIcon,
  rightIcon,
  children,
  className,
  'data-testid': testId,
  'aria-label': ariaLabel,
  ...props
}, ref) => {
  const isDisabled = disabled || loading;
  
  return (
    <ButtonBase
      ref={ref}
      className={className}
      data-testid={testId}
      aria-label={ariaLabel}
      disabled={isDisabled}
      $variant={variant}
      $size={size}
      $color={color}
      $fullWidth={fullWidth}
      $loading={loading}
      {...props}
    >
      {loading && (
        <SpinnerWrapper className="spinner">
          <Spinner size={size === 'sm' ? 'xs' : size === 'lg' ? 'md' : 'sm'} />
        </SpinnerWrapper>
      )}
      
      {leftIcon && !loading && (
        <IconWrapper $position="left">
          {leftIcon}
        </IconWrapper>
      )}
      
      {children}
      
      {rightIcon && !loading && (
        <IconWrapper $position="right">
          {rightIcon}
        </IconWrapper>
      )}
    </ButtonBase>
  );
});

Button.displayName = 'Button';

export default Button;