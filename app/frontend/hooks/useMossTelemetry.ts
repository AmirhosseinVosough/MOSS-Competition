import { useEffect, useMemo, useState } from 'react';
import { RoomEvent } from 'livekit-client';
import { useRoomContext } from '@livekit/components-react';

/**
 * Counts what Moss actually did during this call.
 *
 * The existing `useMossContextEvents` shows *what* was retrieved. This shows
 * *how much* work happened and what it cost, which is the part that makes the
 * architecture visible: retrieval running dozens of times a conversation is
 * only sensible because each one costs milliseconds.
 *
 * Listens for two message types the agent publishes:
 *   moss_context — a retrieval happened
 *   moss_gate    — a sentence was checked before being spoken
 */

const textDecoder = new TextDecoder();

/**
 * What the same work would cost against a hosted vector database.
 *
 * 300ms is the middle of the 200-500ms range Moss cites for a network round
 * trip, and is deliberately the conservative end of what we measured locally
 * (4.77ms including embedding generation).
 */
export const CLOUD_DB_MS = 300;

export type MossTelemetry = {
  /** Retrievals this call. */
  searches: number;
  /** Sentences checked against the care notes before being spoken. */
  gateChecks: number;
  /** Sentences stopped because the notes did not support them. */
  gateBlocks: number;
  /** Median retrieval latency, ms. */
  medianMs: number | null;
  /** Every millisecond spent retrieving this call. */
  totalMs: number;
  /** What the same number of lookups would have cost over a network. */
  cloudEquivalentMs: number;
};

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

export function useMossTelemetry(): MossTelemetry {
  const room = useRoomContext();
  const [durations, setDurations] = useState<number[]>([]);
  const [gateChecks, setGateChecks] = useState(0);
  const [gateBlocks, setGateBlocks] = useState(0);

  useEffect(() => {
    if (!room) return;

    const handleData = (payload: Uint8Array) => {
      let message: { type?: string; data?: Record<string, unknown> };
      try {
        message = JSON.parse(textDecoder.decode(payload));
      } catch {
        return;
      }
      const data = message?.data;
      if (!data || typeof data !== 'object') return;

      const ms = typeof data.time_taken_ms === 'number' ? data.time_taken_ms : null;

      if (message.type === 'moss_context') {
        if (ms !== null) setDurations((prev) => [...prev, ms]);
      } else if (message.type === 'moss_gate') {
        setGateChecks((n) => n + 1);
        if (data.verdict === 'block') setGateBlocks((n) => n + 1);
        if (ms !== null) setDurations((prev) => [...prev, ms]);
      }
    };

    room.on(RoomEvent.DataReceived, handleData);
    return () => {
      room.off(RoomEvent.DataReceived, handleData);
    };
  }, [room]);

  return useMemo(() => {
    const searches = durations.length;
    const totalMs = durations.reduce((sum, d) => sum + d, 0);
    return {
      searches,
      gateChecks,
      gateBlocks,
      medianMs: median(durations),
      totalMs,
      cloudEquivalentMs: searches * CLOUD_DB_MS,
    };
  }, [durations, gateChecks, gateBlocks]);
}
