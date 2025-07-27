import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider } from 'styled-components';
import { theme } from '@/styles/theme';
import Card from './Card';

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('Card', () => {
  it('renders correctly with default props', () => {
    renderWithTheme(
      <Card data-testid="test-card">
        <p>Card content</p>
      </Card>
    );
    
    const card = screen.getByTestId('test-card');
    expect(card).toBeInTheDocument();
    expect(screen.getByText('Card content')).toBeInTheDocument();
  });

  it('renders with different variants', () => {
    const { rerender } = renderWithTheme(
      <Card variant="elevated" data-testid="elevated-card">
        Elevated Card
      </Card>
    );
    expect(screen.getByTestId('elevated-card')).toBeInTheDocument();

    rerender(
      <ThemeProvider theme={theme}>
        <Card variant="outlined" data-testid="outlined-card">
          Outlined Card
        </Card>
      </ThemeProvider>
    );
    expect(screen.getByTestId('outlined-card')).toBeInTheDocument();

    rerender(
      <ThemeProvider theme={theme}>
        <Card variant="filled" data-testid="filled-card">
          Filled Card
        </Card>
      </ThemeProvider>
    );
    expect(screen.getByTestId('filled-card')).toBeInTheDocument();
  });

  it('renders with different padding sizes', () => {
    const { rerender } = renderWithTheme(
      <Card padding="none" data-testid="no-padding">
        No Padding
      </Card>
    );
    expect(screen.getByTestId('no-padding')).toBeInTheDocument();

    rerender(
      <ThemeProvider theme={theme}>
        <Card padding="sm" data-testid="small-padding">
          Small Padding
        </Card>
      </ThemeProvider>
    );
    expect(screen.getByTestId('small-padding')).toBeInTheDocument();

    rerender(
      <ThemeProvider theme={theme}>
        <Card padding="lg" data-testid="large-padding">
          Large Padding
        </Card>
      </ThemeProvider>
    );
    expect(screen.getByTestId('large-padding')).toBeInTheDocument();
  });

  it('handles click events when clickable', () => {
    const handleClick = jest.fn();
    renderWithTheme(
      <Card clickable onClick={handleClick} data-testid="clickable-card">
        Clickable Card
      </Card>
    );
    
    const card = screen.getByTestId('clickable-card');
    fireEvent.click(card);
    
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('does not handle click events when not clickable', () => {
    const handleClick = jest.fn();
    renderWithTheme(
      <Card onClick={handleClick} data-testid="non-clickable-card">
        Non-clickable Card
      </Card>
    );
    
    const card = screen.getByTestId('non-clickable-card');
    fireEvent.click(card);
    
    expect(handleClick).not.toHaveBeenCalled();
  });

  it('supports keyboard navigation when clickable', () => {
    const handleClick = jest.fn();
    renderWithTheme(
      <Card clickable onClick={handleClick} data-testid="keyboard-card">
        Keyboard Card
      </Card>
    );
    
    const card = screen.getByTestId('keyboard-card');
    card.focus();
    
    expect(card).toHaveFocus();
    expect(card).toHaveAttribute('tabIndex', '0');
    expect(card).toHaveAttribute('role', 'button');
    
    fireEvent.keyDown(card, { key: 'Enter' });
    expect(handleClick).toHaveBeenCalledTimes(1);
    
    fireEvent.keyDown(card, { key: ' ' });
    expect(handleClick).toHaveBeenCalledTimes(2);
  });

  it('shows selected state correctly', () => {
    renderWithTheme(
      <Card selected data-testid="selected-card">
        Selected Card
      </Card>
    );
    
    const card = screen.getByTestId('selected-card');
    expect(card).toBeInTheDocument();
  });

  it('applies hoverable styles', () => {
    renderWithTheme(
      <Card hoverable data-testid="hoverable-card">
        Hoverable Card
      </Card>
    );
    
    const card = screen.getByTestId('hoverable-card');
    expect(card).toHaveStyle('cursor: pointer');
  });

  it('forwards ref correctly', () => {
    const ref = React.createRef<HTMLDivElement>();
    renderWithTheme(
      <Card ref={ref} data-testid="ref-card">
        Ref Card
      </Card>
    );
    
    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it('has proper accessibility attributes when clickable', () => {
    renderWithTheme(
      <Card 
        clickable 
        aria-label="Custom card label" 
        data-testid="accessible-card"
      >
        Accessible Card
      </Card>
    );
    
    const card = screen.getByTestId('accessible-card');
    expect(card).toHaveAttribute('aria-label', 'Custom card label');
    expect(card).toHaveAttribute('role', 'button');
    expect(card).toHaveAttribute('tabIndex', '0');
  });

  it('prevents default behavior on space key press', () => {
    const handleClick = jest.fn();
    renderWithTheme(
      <Card clickable onClick={handleClick} data-testid="space-card">
        Space Card
      </Card>
    );
    
    const card = screen.getByTestId('space-card');
    const spaceEvent = new KeyboardEvent('keydown', { key: ' ' });
    const preventDefaultSpy = jest.spyOn(spaceEvent, 'preventDefault');
    
    fireEvent.keyDown(card, spaceEvent);
    
    expect(preventDefaultSpy).toHaveBeenCalled();
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});