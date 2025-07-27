import React, { forwardRef } from 'react';
import styled, { css } from 'styled-components';
import { ListItemProps } from '../types';
import { 
  getSpacing, 
  getColor, 
  getTypography,
  getBorderRadius,
  getAnimation,
  getFocusRing
} from '../utils';

const ListItemBase = styled.div<{
  $active: boolean;
  $disabled: boolean;
  $clickable: boolean;
  $divider: boolean;
}>`
  /* Base styles */
  display: flex;
  align-items: center;
  gap: ${getSpacing(3)};
  padding: ${getSpacing(3)} ${getSpacing(4)};
  min-height: 48px; /* Following 8pt grid - 6 * 8pt */
  position: relative;
  
  /* Animation */
  ${getAnimation('all', '150')}
  
  /* Clickable styles */
  ${({ $clickable }) => $clickable && css`
    cursor: pointer;
    user-select: none;
    
    &:hover:not([aria-disabled="true"]) {
      background-color: ${getColor('neutral.50')};
    }
    
    &:active:not([aria-disabled="true"]) {
      background-color: ${getColor('neutral.100')};
    }
  `}
  
  /* Active state */
  ${({ $active }) => $active && css`
    background-color: ${getColor('primary.50')};
    color: ${getColor('primary.700')};
    
    &::before {
      content: '';
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 3px;
      background-color: ${getColor('primary.500')};
    }
    
    [dir="rtl"] &::before {
      left: auto;
      right: 0;
    }
  `}
  
  /* Disabled state */
  ${({ $disabled }) => $disabled && css`
    opacity: 0.6;
    cursor: not-allowed;
    pointer-events: none;
    color: ${getColor('text.disabled')};
  `}
  
  /* Divider */
  ${({ $divider }) => $divider && css`
    border-bottom: 1px solid ${getColor('border.secondary')};
  `}
  
  /* Focus styles */
  ${({ $clickable }) => $clickable && css`
    &:focus-visible {
      ${getFocusRing(getColor('primary.500'))}
    }
  `}
  
  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    transition: none;
  }
`;

const LeftContent = styled.div`
  display: flex;
  align-items: center;
  flex-shrink: 0;
`;

const MainContent = styled.div`
  flex: 1;
  min-width: 0; /* Allow text truncation */
  display: flex;
  flex-direction: column;
  gap: ${getSpacing(1)};
`;

const PrimaryText = styled.div`
  font-size: ${getTypography('fontSize', 'base')};
  font-weight: ${getTypography('fontWeight', 'medium')};
  line-height: ${getTypography('lineHeight', 'tight')};
  color: inherit;
  
  /* Text truncation */
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const SecondaryText = styled.div`
  font-size: ${getTypography('fontSize', 'sm')};
  font-weight: ${getTypography('fontWeight', 'normal')};
  line-height: ${getTypography('lineHeight', 'normal')};
  color: ${getColor('text.secondary')};
  
  /* Text truncation */
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const TertiaryText = styled.div`
  font-size: ${getTypography('fontSize', 'xs')};
  font-weight: ${getTypography('fontWeight', 'normal')};
  line-height: ${getTypography('lineHeight', 'normal')};
  color: ${getColor('text.tertiary')};
  
  /* Text truncation */
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const RightContent = styled.div`
  display: flex;
  align-items: center;
  gap: ${getSpacing(2)};
  flex-shrink: 0;
`;

const AvatarWrapper = styled.div`
  display: flex;
  align-items: center;
  margin-right: ${getSpacing(1)};
  
  [dir="rtl"] & {
    margin-right: 0;
    margin-left: ${getSpacing(1)};
  }
`;

const IconWrapper = styled.div`
  display: flex;
  align-items: center;
  color: ${getColor('text.secondary')};
  
  svg {
    width: 20px;
    height: 20px;
  }
`;

const ListItem = forwardRef<HTMLDivElement, ListItemProps>(({
  primary,
  secondary,
  tertiary,
  leftIcon,
  rightIcon,
  avatar,
  badge,
  active = false,
  disabled = false,
  divider = false,
  clickable = false,
  children,
  className,
  'data-testid': testId,
  onClick,
  onKeyDown,
  tabIndex,
  role,
  'aria-label': ariaLabel,
  'aria-disabled': ariaDisabled,
  ...props
}, ref) => {
  const isClickable = clickable || !!onClick;
  const isDisabled = disabled || ariaDisabled;

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (isClickable && onClick && !isDisabled && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      onClick(event as any);
    }
    onKeyDown?.(event);
  };

  const listItemProps = {
    ref,
    className,
    'data-testid': testId,
    onClick: isClickable && !isDisabled ? onClick : undefined,
    onKeyDown: isClickable ? handleKeyDown : onKeyDown,
    tabIndex: isClickable && !isDisabled ? (tabIndex ?? 0) : tabIndex,
    role: isClickable ? (role ?? 'button') : role,
    'aria-label': ariaLabel,
    'aria-disabled': isDisabled ? 'true' : 'false',
    $active: active,
    $disabled: !!isDisabled,
    $clickable: isClickable,
    $divider: divider,
    ...props,
  };

  return (
    <ListItemBase {...listItemProps}>
      <LeftContent>
        {avatar && (
          <AvatarWrapper>
            {avatar}
          </AvatarWrapper>
        )}
        {leftIcon && (
          <IconWrapper>
            {leftIcon}
          </IconWrapper>
        )}
      </LeftContent>
      
      <MainContent>
        <PrimaryText>{primary}</PrimaryText>
        {secondary && <SecondaryText>{secondary}</SecondaryText>}
        {tertiary && <TertiaryText>{tertiary}</TertiaryText>}
        {children}
      </MainContent>
      
      <RightContent>
        {badge}
        {rightIcon && (
          <IconWrapper>
            {rightIcon}
          </IconWrapper>
        )}
      </RightContent>
    </ListItemBase>
  );
});

ListItem.displayName = 'ListItem';

export default ListItem;