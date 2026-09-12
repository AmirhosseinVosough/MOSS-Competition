'use client';

import * as React from 'react';
import { CLOUD_DB_MS, useMossTelemetry } from '@/hooks/useMossTelemetry';
import { cn } from '@/lib/shadcn/utils';

/**
 * Live counter of what Moss did during this call.
 *
 * The point is not that retrieval is fast -- a number in a document can say
 * that. The point is that it runs constantly, and the comparison line is what
 * makes it land: the same lookups against a hosted vector database would have
 * cost seconds, which is more than a conversation has to spend.
 */

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-foreground font-mono text-2xl leading-none font-semibold tabular-nums">
        {value}
      </span>
      <span className="text-muted-foreground mt-1 text-xs tracking-wide uppercase">{label}</span>
    </div>
  );
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function MossLatencyHud({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  const { searches, gateChecks, gateBlocks, medianMs, totalMs, cloudEquivalentMs } =
    useMossTelemetry();

  // Nothing to report until the first lookup lands.
  if (searches === 0) return null;

  return (
    <div
      className={cn('border-border bg-card rounded-xl border p-4 shadow-sm', className)}
      {...props}
    >
      <h3 className="text-muted-foreground mb-3 text-xs font-medium tracking-wide uppercase">
        Moss · this call
      </h3>

      <div className="flex flex-wrap gap-x-8 gap-y-4">
        <Stat value={String(searches)} label="lookups" />
        <Stat value={medianMs === null ? '—' : `${medianMs.toFixed(1)}`} label="median ms" />
        <Stat value={formatMs(totalMs)} label="total spent" />
      </div>

      {gateChecks > 0 && (
        <p className="text-muted-foreground mt-4 text-sm">
          <span className="text-foreground font-semibold tabular-nums">{gateChecks}</span>{' '}
          {gateChecks === 1 ? 'sentence' : 'sentences'} checked against the care notes before being
          spoken
          {gateBlocks > 0 && (
            <>
              {' — '}
              <span className="font-semibold text-amber-600 dark:text-amber-500">
                {gateBlocks} stopped
              </span>
            </>
          )}
          .
        </p>
      )}

      <div className="border-border mt-4 border-t pt-3">
        <p className="text-muted-foreground text-sm leading-relaxed">
          The same {searches} {searches === 1 ? 'lookup' : 'lookups'} against a hosted vector
          database at {CLOUD_DB_MS} ms would have taken{' '}
          <span className="text-foreground font-semibold tabular-nums">
            {formatMs(cloudEquivalentMs)}
          </span>
          .
        </p>
      </div>
    </div>
  );
}
