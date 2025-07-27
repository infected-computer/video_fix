import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider } from 'styled-components';
import { theme } from '@/styles/theme';
import Button from './Button';

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('Button', () => {
  it('renders correctly with default props', () => {
    renderWithTheme(<Button>Click me</Button>);
    
    const button = screen.getByRole('button', { name: 'Click me' });
    expect(button).toBeInTheDocument();
    expect(button).not.toBeDisabled();
  });

  it('renders with different variants', () => {
    const { rerender } = renderWithTheme(<Button variant="primary">Primary</Button>);
    expect(screen.getByRole('button')).toBeInTheDocument();

    rerender(
      <ThemeProvider theme={theme}>
        <Button variant="secondary">Secondary</Button>
      </ThemeProvider>
    );
    expect(screen.getByRole('button')).toBeInTheDocument();

    rerender(
      <ThemeProvider theme={theme}>
        <Button variant="tertiary">Tertiary</Button>
      </ThemeProvider>
    );
    expect(screen.getByRole('button')).toBeInTheDocument();
  });

  it('renders with different sizes', () => {
    const { rerender } = renderWithTheme(<Button size="sm">Small</Button>);
    expect(screen.getByRole('button')).toBeInTheDocument();

    rerender(
      <ThemeProvider theme={theme}>
        <Button size="md">Medium</Button>
      </ThemeProvider>
    );
    expect(screen.getByRole('button')).toBeInTheDocument();

    rerender(
      <ThemeProvider theme={theme}>
        <Button size="lg">Large</Button>
      </ThemeProvider>
    );
    expect(screen.getByRole('button')).toBeInTheDocument();
  });

  it('handles click events', () => {
    const handleClick = jest.fn();
    renderWithTheme(<Button onClick={handleClick}>Click me</Button>);
    
    const button = screen.getByRole('button');
    fireEvent.click(button);
    
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('is disabled when disabled prop is true', () => {
    renderWithTheme(<Button disabled>Disabled</Button>);
    
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
  });

  it('is disabled when loading prop is true', () => {
    renderWithTheme(<Button loading>Loading</Button>);
    
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
  });

  it('does not call onClick when disabled', () => {
    const handleClick = jest.fn();
    renderWithTheme(<Button disabled onClick={handleClick}>Disabled</Button>);
    
    const button = screen.getByRole('button');
    fireEvent.click(button);
    
    expect(handleClick).not.toHaveBeenCalled();
  });

  it('renders with left icon', () => {
    const LeftIcon = () => <span data-testid="left-icon">←</span>;
    renderWithTheme(
      <Button leftIcon={<LeftIcon />}>With Left Icon</Button>
    );
    
    expect(screen.getByTestId('left-icon')).toBeInTheDocument();
  });

  it('renders with right icon', () => {
    const RightIcon = () => <span data-testid="right-icon">→</span>;
    renderWithTheme(
      <Button rightIcon={<RightIcon />}>With Right Icon</Button>
    );
    
    expect(screen.getByTestId('right-icon')).toBeInTheDocument();
  });

  it('renders full width when fullWidth prop is true', () => {
    renderWithTheme(<Button fullWidth>Full Width</Button>);
    
    const button = screen.getByRole('button');
    expect(button).toHaveStyle('width: 100%');
  });

  it('has proper accessibility attributes', () => {
    renderWithTheme(
      <Button aria-label="Custom label" data-testid="test-button">
        Button
      </Button>
    );
    
    const button = screen.getByTestId('test-button');
    expect(button).toHaveAttribute('aria-label', 'Custom label');
  });

  it('supports keyboard navigation', () => {
    const handleClick = jest.fn();
    renderWithTheme(<Button onClick={handleClick}>Keyboard Test</Button>);
    
    const button = screen.getByRole('button');
    button.focus();
    
    expect(button).toHaveFocus();
    
    fireEvent.keyDown(button, { key: 'Enter' });
    expect(handleClick).toHaveBeenCalledTimes(1);
    
    fireEvent.keyDown(button, { key: ' ' });
    expect(handleClick).toHaveBeenCalledTimes(2);
  });

  it('renders with different colors', () => {
    const { rerender } = renderWithTheme(<Button color="primary">Primary Color</Button>);
    expect(screen.getByRole('button')).toBeInTheDocument();

    rerender(
      <ThemeProvider theme={theme}>
        <Button color="success">Success Color</Button>
      </ThemeProvider>
    );
    expect(screen.getByRole('button')).toBeInTheDocument();

    rerender(
      <ThemeProvider theme={theme}>
        <Button color="error">Error Color</Button>
      </ThemeProvider>
    );
    expect(screen.getByRole('button')).toBeInTheDocument();
  });

  it('forwards ref correctly', () => {
    const ref = React.createRef<HTMLButtonElement>();
    renderWithTheme(<Button ref={ref}>Ref Test</Button>);
    
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });
});