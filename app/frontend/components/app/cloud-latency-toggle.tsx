'use client';

import * as React from 'react';
import { useRoomContext } from '@livekit/components-react';
import { cn } from '@/lib/shadcn/utils';

/**
 * Demo switch: make every Moss lookup pretend to cross a network.
 *
 * The agent's retrieval is fast enough that the architecture is invisible when
 * it works -- nothing looks impressive about a conversation that simply feels
 * normal. This puts the handicap back so the difference can be heard rather
 * than asserted: same agent, same question, 300ms per lookup instead of ~6ms.
 *
 * Deliberately a manual switch rather than something the app decides. It exists
 * to be flipped in front of someone mid-conversation.
 */

const encoder = new TextEncoder();

export function CloudLatencyToggle({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  const room = useRoomContext();
  const [enabled, setEnabled] = React.useState(false);

  const toggle = React.useCallback(async () => {
    const next = !enabled;
    setEnabled(next);
    try {
      await room?.localParticipant?.publishData(
        encoder.encode(JSON.stringify({ type: 'moss_simulate_cloud', enabled: next })),
        { reliable: true }
      );
    } catch (error) {
      // Put the switch back if the agent never heard us, rather than showing a
      // state the agent is not actually in.
      console.warn('could not reach the agent', error);
      setEnabled(!next);
    }
  }, [enabled, room]);

  return (
    <div
      className={cn(
        'rounded-xl border p-3 transition-colors',
        enabled ? 'border-amber-500 bg-amber-500/10' : 'border-border bg-card',
        className
      )}
      {...props}
    >
      <label className="flex cursor-pointer items-start gap-3">
        <input
          type="checkbox"
          checked={enabled}
          onChange={toggle}
          className="mt-0.5 size-5 shrink-0 cursor-pointer accent-amber-600"
        />
        <span className="min-w-0">
          <span className="text-foreground block text-sm font-semibold">
            Simulate a hosted vector database
          </span>
          <span className="text-muted-foreground block text-xs leading-snug">
            {enabled
              ? 'On — every lookup now waits 300 ms, as it would over a network.'
              : 'Off — lookups run in-process, about 6 ms.'}
          </span>
        </span>
      </label>
    </div>
  );
}
