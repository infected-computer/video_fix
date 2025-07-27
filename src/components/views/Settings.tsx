import React from 'react';
import styled from 'styled-components';
import { Card } from '@/components/design-system';

const SettingsContainer = styled.div`
  padding: 24px;
  height: 100%;
  overflow-y: auto;
`;

const Settings: React.FC = () => {
  return (
    <SettingsContainer>
      <Card>
        <h2>הגדרות</h2>
        <p>כאן יהיו הגדרות האפליקציה</p>
      </Card>
    </SettingsContainer>
  );
};

export default Settings;