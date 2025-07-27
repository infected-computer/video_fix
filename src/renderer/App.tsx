import React, { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import styled from 'styled-components';

import { useAppDispatch } from '@/hooks/redux';
import { initializeApp } from '@/store/slices/appSlice';
import MainLayout from '@/components/layout/MainLayout';
import Dashboard from '@/components/views/Dashboard';
import VideoRepair from '@/components/views/VideoRepair';
import VideoRecovery from '@/components/views/VideoRecovery';
import FileAnalysis from '@/components/views/FileAnalysis';
import Settings from '@/components/views/Settings';
import ErrorBoundary from '@/components/common/ErrorBoundary';

const AppContainer = styled.div`
  height: 100vh;
  width: 100vw;
  display: flex;
  flex-direction: column;
  background-color: ${({ theme }) => theme.colors.background.primary};
  color: ${({ theme }) => theme.colors.text.primary};
  font-family: ${({ theme }) => theme.typography.fontFamily.primary};
`;

const App: React.FC = () => {
  const dispatch = useAppDispatch();

  useEffect(() => {
    // Initialize the application
    dispatch(initializeApp());
  }, [dispatch]);

  return (
    <ErrorBoundary>
      <AppContainer>
        <MainLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/repair" element={<VideoRepair />} />
            <Route path="/recovery" element={<VideoRecovery />} />
            <Route path="/analysis" element={<FileAnalysis />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </MainLayout>
      </AppContainer>
    </ErrorBoundary>
  );
};

export default App;