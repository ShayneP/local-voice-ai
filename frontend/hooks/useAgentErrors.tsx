import { useEffect } from 'react';
import { useAgent, useSessionContext } from '@livekit/components-react';
import { toastAlert } from '@/components/livekit/alert-toast';

export function useAgentErrors() {
  const agent = useAgent();
  const { isConnected, end } = useSessionContext();

  useEffect(() => {
    if (isConnected && agent.state === 'failed') {
      const reasons = agent.failureReasons;
      const description = reasons.length > 1 ? reasons.join(' / ') : undefined;
      const title = reasons.length === 1 ? reasons[0] : 'セッションエラー';

      toastAlert({ title, description });
      end();
    }
  }, [agent, isConnected, end]);
}
