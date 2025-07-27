import React from 'react';
import styled from 'styled-components';
import { Card } from '@/components/design-system';

const RecoveryContainer = styled.div`
  padding: 24px;
  height: 100%;
  overflow-y: auto;
`;

const VideoRecovery: React.FC = () => {
  return (
    <RecoveryContainer>
      <Card>
        <h2>שחזור וידיאו</h2>
        <p>כאן יהיה ממשק שחזור הוידיאו</p>
      </Card>
    </RecoveryContainer>
  );
};

export default VideoRecovery;