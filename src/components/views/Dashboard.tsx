import React from 'react';
import styled from 'styled-components';
import { Card, Button } from '@/components/design-system';

const DashboardContainer = styled.div`
  padding: 24px;
  height: 100%;
  overflow-y: auto;
`;

const WelcomeCard = styled(Card)`
  margin-bottom: 24px;
`;

const Dashboard: React.FC = () => {
  return (
    <DashboardContainer>
      <WelcomeCard>
        <h1>ברוכים הבאים ל-PhoenixDRS Professional</h1>
        <p>מערכת שחזור וידיאו מתקדמת</p>
        <Button variant="primary">התחל עכשיו</Button>
      </WelcomeCard>
    </DashboardContainer>
  );
};

export default Dashboard;