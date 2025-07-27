import React, { forwardRef } from 'react';
import styled, { css } from 'styled-components';
import { CardProps } from '../types';
import { 
  getSpacing, 
  getColor, 
  getBorderRadius, 
  getShadow,
  getAnimation,
  getFocusRing
} from '../utils';

const CardBase = styled.div<{
  $variant: CardProps['variant'];
  $padding: CardProps['padding'];
  $hoverable: boolean;
  $clickable: boolean;
  $selected: boolean;
}>`
  /* Base styles */
  background-color: ${getColor('surface.primary')};
  border-radius: ${getBorderRadius('base')};
  position: relative;
  overflow: hidden;
  
  /* Animation */
  ${getAnimation('all', '200')}
  
  /* Variant styles */
  ${({ $variant }) => {
    switch ($variant) {
      case 'elevated':
        return css`
          box-shadow: ${getShadow('base')};
          border: none;
        `;
        
      case 'outlined':
        return css`
          border: 1px solid ${getColor('border.primary')};
          box-shadow: none;
        `;
        
      case 'filled':
        return css`
          background-color: ${getColor('surface.secondary')};
          border: none;
          box-shadow: none;
        `;
        
      default:
        return css`
          box-shadow: ${getShadow('base')};
          border: none;
        `;
    }
  }}
  
  /* Padding styles */
  ${({ $padding }) => {
    switch ($padding) {
      case 'none':
        return css`padding: 0;`;
      case 'sm':
        return css`padding: ${getSpacing(3)};`;
      case 'md':
        return css`padding: ${getSpacing(4)};`;
      case 'lg':
        return css`padding: ${getSpacing(6)};`;
      default:
        return css`padding: ${getSpacing(4)};`;
    }
  }}
  
  /* Hoverable styles */
  ${({ $hoverable, $variant }) => $hoverable && css`
    cursor: pointer;
    
    &:hover {
      ${$variant === 'elevated' && css`
        box-shadow: ${getShadow('md')};
        transform: translateY(-2px);
      `}
      
      ${$variant === 'outlined' && css`
        border-color: ${getColor('border.focus')};
        box-shadow: ${getShadow('sm')};
      `}
      
      ${$variant === 'filled' && css`
        background-color: ${getColor('surface.tertiary')};
      `}
    }
  `}
  
  /* Clickable styles */
  ${({ $clickable }) => $clickable && css`
    cursor: pointer;
    user-select: none;
    
    &:active {
      transform: scale(0.98);
    }
  `}
  
  /* Selected styles */
  ${({ $selected, $variant }) => $selected && css`
    ${$variant === 'elevated' && css`
      box-shadow: ${getShadow('lg')};
      border: 2px solid ${getColor('primary.500')};
    `}
    
    ${$variant === 'outlined' && css`
      border-color: ${getColor('primary.500')};
      border-width: 2px;
      box-shadow: 0 0 0 1px ${getColor('primary.500')};
    `}
    
    ${$variant === 'filled' && css`
      background-color: ${getColor('primary.50')};
      border: 2px solid ${getColor('primary.500')};
    `}
  `}
  
  /* Focus styles for clickable cards */
  ${({ $clickable }) => $clickable && css`
    &:focus-visible {
      ${getFocusRing(getColor('primary.500'))}
    }
  `}
  
  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    transition: none;
    transform: none !important;
  }
  
  /* High contrast mode */
  @media (prefers-contrast: high) {
    border-width: 2px;
    
    ${({ $variant }) => $variant === 'elevated' && css`
      border: 2px solid ${getColor('border.primary')};
    `}
  }
`;

const Card = forwardRef<HTMLDivElement, CardProps>(({
  variant = 'elevated',
  padding = 'md',
  hoverable = false,
  clickable = false,
  selected = false,
  children,
  className,
  'data-testid': testId,
  onClick,
  onKeyDown,
  tabIndex,
  role,
  'aria-label': ariaLabel,
  ...props
}, ref) => {
  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (clickable && onClick && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      onClick(event as any);
    }
    onKeyDown?.(event);
  };

  const cardProps = {
    ref,
    className,
    'data-testid': testId,
    onClick: clickable ? onClick : undefined,
    onKeyDown: clickable ? handleKeyDown : onKeyDown,
    tabIndex: clickable ? (tabIndex ?? 0) : tabIndex,
    role: clickable ? (role ?? 'button') : role,
    'aria-label': ariaLabel,
    $variant: variant,
    $padding: padding,
    $hoverable: hoverable,
    $clickable: clickable,
    $selected: selected,
    ...props,
  };

  return (
    <CardBase {...cardProps}>
      {children}
    </CardBase>
  );
});

Card.displayName = 'Card';

export default Card;