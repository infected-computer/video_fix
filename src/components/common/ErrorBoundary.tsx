import React, { Component, ErrorInfo, ReactNode } from 'react';
import styled from 'styled-components';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

const ErrorContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  padding: 24px;
  text-align: center;
`;

const ErrorTitle = styled.h1`
  color: #dc3545;
  margin-bottom: 16px;
`;

const ErrorMessage = styled.p`
  color: #6c757d;
  margin-bottom: 24px;
`;

const ReloadButton = styled.button`
  background-color: #007acc;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  
  &:hover {
    background-color: #005a9e;
  }
`;

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
  }

  private handleReload = () => {
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <ErrorContainer>
          <ErrorTitle>משהו השתבש</ErrorTitle>
          <ErrorMessage>
            אירעה שגיאה בלתי צפויה. אנא נסה לרענן את הדף.
          </ErrorMessage>
          <ReloadButton onClick={this.handleReload}>
            רענן דף
          </ReloadButton>
        </ErrorContainer>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;