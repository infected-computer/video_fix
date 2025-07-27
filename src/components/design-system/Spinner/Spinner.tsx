import React from 'react';
import styled, { keyframes } from 'styled-components';
import { SpinnerProps } from '../types';
import { getColor, getSpacing } from '../utils';

const spin = keyframes`
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
`;

const SpinnerBase = styled.div<{
  $size: SpinnerProps['size'];
  $color: SpinnerProps['color'];
  $thickness: number;
}>`
  display: inline-block;
  border-radius: 50%;
  border-style: solid;
  border-color: transparent;
  animation: ${spin} 1s linear infinite;
  
  /* Size styles */
  ${({ $size, $thickness }) => {
    const sizeMap = {
      xs: { size: '16px', thickness: $thickness || 2 },
      sm: { size: '20px', thickness: $thickness || 2 },
      md: { size: '24px', thickness: $thickness || 3 },
      lg: { size: '32px', thickness: $thickness || 3 },
      xl: { size: '40px', thickness: $thickness || 4 },
    };
    
    const config = sizeMap[$size || 'md'];
    
    return `
      width: ${config.size};
      height: ${config.size};
      border-width: ${config.thickness}px;
    `;
  }}
  
  /* Color styles */
  ${({ $color }) => {
    const color = $color || 'primary';
    return `
      border-top-color: ${getColor(`${color}.500`)};
      border-right-color: ${getColor(`${color}.200`)};
    `;
  }}
  
  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    animation: none;
    border-top-color: ${({ $color }) => getColor(`${$color || 'primary'}.500`)};
    border-right-color: ${({ $color }) => getColor(`${$color || 'primary'}.500`)};
    border-bottom-color: ${({ $color }) => getColor(`${$color || 'primary'}.500`)};
    border-left-color: ${({ $color }) => getColor(`${$color || 'primary'}.500`)};
  }
`;

const Spinner: React.FC<SpinnerProps> = ({
  size = 'md',
  color = 'primary',
  thickness,
  className,
  'data-testid': testId,
  ...props
}) => {
  return (
    <SpinnerBase
      className={className}
      data-testid={testId}
      $size={size}
      $color={color}
      $thickness={thickness || 0}
      role="status"
      aria-label="Loading"
      {...props}
    />
  );
};

export default Spinner;