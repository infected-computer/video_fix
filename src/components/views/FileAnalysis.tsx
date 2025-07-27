import React from 'react';
import styled from 'styled-components';
import { Card } from '@/components/design-system';

const AnalysisContainer = styled.div`
  padding: 24px;
  height: 100%;
  overflow-y: auto;
`;

const FileAnalysis: React.FC = () => {
  return (
    <AnalysisContainer>
      <Card>
        <h2>ניתוח קבצים</h2>
        <p>כאן יהיה ממשק ניתוח הקבצים</p>
      </Card>
    </AnalysisContainer>
  );
};

export default FileAnalysis;