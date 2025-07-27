import React, { forwardRef, useState, useCallback } from 'react';
import styled, { css } from 'styled-components';
import { SidebarProps } from '../types';
import { 
  getSpacing, 
  getColor, 
  getShadow,
  getAnimation,
  getZIndex
} from '../utils';
import ListItem from '../ListItem';
import Button from '../Button';

const SidebarBase = styled.aside<{
  $collapsed: boolean;
  $width: number;
}>`
  /* Base styles */
  display: flex;
  flex-direction: column;
  background-color: ${getColor('surface.primary')};
  border-right: 1px solid ${getColor('border.primary')};
  height: 100%;
  position: relative;
  z-index: ${getZIndex('docked')};
  
  /* Width and collapse animation */
  width: ${({ $collapsed, $width }) => $collapsed ? '64px' : `${$width}px`};
  ${getAnimation('width', '300')}
  
  /* RTL support */
  [dir="rtl"] & {
    border-right: none;
    border-left: 1px solid ${getColor('border.primary')};
  }
  
  /* Shadow for elevated appearance */
  box-shadow: ${getShadow('sm')};
  
  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    transition: none;
  }
`;

const SidebarHeader = styled.div<{ $collapsed: boolean }>`
  padding: ${getSpacing(4)};
  border-bottom: 1px solid ${getColor('border.secondary')};
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 64px; /* 8 * 8pt */
  
  ${({ $collapsed }) => $collapsed && css`
    padding: ${getSpacing(4)} ${getSpacing(2)};
    justify-content: center;
  `}
`;

const SidebarContent = styled.div`
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: ${getSpacing(2)} 0;
`;

const SidebarFooter = styled.div<{ $collapsed: boolean }>`
  padding: ${getSpacing(4)};
  border-top: 1px solid ${getColor('border.secondary')};
  
  ${({ $collapsed }) => $collapsed && css`
    padding: ${getSpacing(4)} ${getSpacing(2)};
  `}
`;

const CollapseButton = styled(Button)`
  min-width: 32px;
  width: 32px;
  height: 32px;
  padding: 0;
  
  svg {
    width: 16px;
    height: 16px;
  }
`;

const SidebarGroup = styled.div`
  margin-bottom: ${getSpacing(4)};
`;

const SidebarGroupTitle = styled.div<{ $collapsed: boolean }>`
  padding: ${getSpacing(2)} ${getSpacing(4)};
  font-size: ${getColor('typography.fontSize.xs')};
  font-weight: ${getColor('typography.fontWeight.semibold')};
  color: ${getColor('text.tertiary')};
  text-transform: uppercase;
  letter-spacing: 0.05em;
  
  ${({ $collapsed }) => $collapsed && css`
    opacity: 0;
    pointer-events: none;
  `}
  
  ${getAnimation('opacity', '200')}
`;

const SidebarItemWrapper = styled.div<{ $collapsed: boolean; $hasChildren: boolean }>`
  position: relative;
  
  ${({ $hasChildren }) => $hasChildren && css`
    .sidebar-item {
      padding-right: ${getSpacing(6)};
      
      [dir="rtl"] & {
        padding-right: ${getSpacing(4)};
        padding-left: ${getSpacing(6)};
      }
    }
  `}
`;

const ExpandButton = styled.button<{ $expanded: boolean }>`
  position: absolute;
  right: ${getSpacing(2)};
  top: 50%;
  transform: translateY(-50%);
  width: 24px;
  height: 24px;
  border: none;
  background: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: ${getColor('borderRadius.sm')};
  color: ${getColor('text.secondary')};
  
  ${getAnimation('transform', '150')}
  
  &:hover {
    background-color: ${getColor('neutral.100')};
  }
  
  &:focus-visible {
    outline: 2px solid ${getColor('primary.500')};
    outline-offset: 1px;
  }
  
  ${({ $expanded }) => $expanded && css`
    transform: translateY(-50%) rotate(90deg);
  `}
  
  [dir="rtl"] & {
    right: auto;
    left: ${getSpacing(2)};
    
    ${({ $expanded }) => $expanded && css`
      transform: translateY(-50%) rotate(-90deg);
    `}
  }
  
  svg {
    width: 12px;
    height: 12px;
  }
`;

const SubItemsList = styled.div<{ $expanded: boolean; $collapsed: boolean }>`
  overflow: hidden;
  max-height: ${({ $expanded }) => $expanded ? '1000px' : '0'};
  ${getAnimation('max-height', '300')}
  
  ${({ $collapsed }) => $collapsed && css`
    display: none;
  `}
  
  .sidebar-item {
    padding-left: ${getSpacing(8)};
    
    [dir="rtl"] & {
      padding-left: ${getSpacing(4)};
      padding-right: ${getSpacing(8)};
    }
  }
`;

const Sidebar = forwardRef<HTMLElement, SidebarProps>(({
  items,
  activeKey,
  collapsed = false,
  collapsible = true,
  width = 240,
  onItemClick,
  onCollapse,
  children,
  className,
  'data-testid': testId,
  ...props
}, ref) => {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

  const handleItemClick = useCallback((key: string, item: any) => {
    if (item.children && item.children.length > 0) {
      setExpandedItems(prev => {
        const newSet = new Set(prev);
        if (newSet.has(key)) {
          newSet.delete(key);
        } else {
          newSet.add(key);
        }
        return newSet;
      });
    } else {
      onItemClick?.(key, item);
    }
  }, [onItemClick]);

  const handleCollapseToggle = useCallback(() => {
    onCollapse?.(!collapsed);
  }, [collapsed, onCollapse]);

  const renderItem = (item: any, level = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedItems.has(item.key);
    const isActive = activeKey === item.key;

    return (
      <SidebarItemWrapper 
        key={item.key} 
        $collapsed={collapsed} 
        $hasChildren={hasChildren}
      >
        <ListItem
          className="sidebar-item"
          primary={collapsed && level === 0 ? '' : item.label}
          leftIcon={item.icon}
          rightIcon={item.badge}
          active={isActive}
          disabled={item.disabled}
          clickable={!item.disabled}
          onClick={() => handleItemClick(item.key, item)}
          aria-label={collapsed ? item.label : undefined}
          title={collapsed ? item.label : undefined}
        />
        
        {hasChildren && !collapsed && (
          <>
            <ExpandButton
              $expanded={isExpanded}
              onClick={(e) => {
                e.stopPropagation();
                handleItemClick(item.key, item);
              }}
              aria-label={`${isExpanded ? 'Collapse' : 'Expand'} ${item.label}`}
            >
              <svg viewBox="0 0 12 12" fill="currentColor">
                <path d="M4.5 3L7.5 6L4.5 9" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </ExpandButton>
            
            <SubItemsList $expanded={isExpanded} $collapsed={collapsed}>
              {item.children.map((child: any) => renderItem(child, level + 1))}
            </SubItemsList>
          </>
        )}
      </SidebarItemWrapper>
    );
  };

  return (
    <SidebarBase
      ref={ref}
      className={className}
      data-testid={testId}
      $collapsed={collapsed}
      $width={width}
      {...props}
    >
      <SidebarHeader $collapsed={collapsed}>
        {!collapsed && (
          <div>
            {/* Logo or title can go here */}
          </div>
        )}
        
        {collapsible && (
          <CollapseButton
            variant="ghost"
            size="sm"
            onClick={handleCollapseToggle}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <svg viewBox="0 0 16 16" fill="currentColor">
              {collapsed ? (
                <path d="M6 3L10 8L6 13" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
              ) : (
                <path d="M10 3L6 8L10 13" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
              )}
            </svg>
          </CollapseButton>
        )}
      </SidebarHeader>
      
      <SidebarContent>
        {items.map(item => renderItem(item))}
        {children}
      </SidebarContent>
      
      {!collapsed && (
        <SidebarFooter $collapsed={collapsed}>
          {/* Footer content can go here */}
        </SidebarFooter>
      )}
    </SidebarBase>
  );
});

Sidebar.displayName = 'Sidebar';

export default Sidebar;