import React from 'react';
import styled from 'styled-components';
import { Card } from '@/components/design-system';

const RepairContainer = styled.div`
  padding: 24px;
  height: 100%;
  overflow-y: auto;
`;

const VideoRepair: React.FC = () => {
  return (
    <RepairContainer>
      <Card>
        <h2>תיקון וידיאו</h2>
        <p>כאן יהיה ממשק תיקון הוידיאו</p>
      </Card>
    </RepairContainer>
  );
};

export default VideoRepair;