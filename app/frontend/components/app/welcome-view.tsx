import { Button } from '@/components/ui/button';

/**
 * The first thing Bill sees.
 *
 * Designed for an older adult rather than a developer, which drives every
 * choice here: body text at 20px and above (the starter used 12px), a button
 * large enough to hit without precision, wording with no jargon in it, and no
 * links away to documentation.
 *
 * The example questions are not decoration. Someone who has never used a voice
 * assistant does not know what it is allowed to be asked, and silence in front
 * of a microphone is the most common way these products fail on first contact.
 */

function SuggestionCard({ children }: { children: React.ReactNode }) {
  return (
    <li className="border-border bg-card rounded-2xl border-2 px-5 py-4 text-left text-lg leading-snug">
      <span aria-hidden="true" className="mr-2 select-none">
        &ldquo;
      </span>
      {children}
      <span aria-hidden="true" className="select-none">
        &rdquo;
      </span>
    </li>
  );
}

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: () => void;
}

export const WelcomeView = ({
  startButtonText,
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  return (
    <div ref={ref} className="flex min-h-svh w-full items-center justify-center px-6 py-10">
      <section className="flex w-full max-w-xl flex-col items-center text-center">
        <h1 className="text-foreground text-4xl leading-tight font-semibold tracking-tight sm:text-5xl">
          Hello, Bill
        </h1>

        <p className="text-foreground mt-5 max-w-md text-xl leading-relaxed">
          I can help with your medicines, your appointments, and anything Sarah has left for you.
        </p>

        {/* A large, unambiguous target. The starter's button was 12px uppercase
            monospace, which is close to unreadable for the intended user. */}
        <Button
          size="lg"
          onClick={onStartCall}
          className="mt-10 h-auto w-full max-w-md rounded-full px-8 py-7 text-2xl font-semibold shadow-lg"
        >
          {startButtonText}
        </Button>

        <p className="text-muted-foreground mt-4 text-base">
          Then just speak normally. Take your time.
        </p>

        <div className="mt-12 w-full">
          <h2 className="text-muted-foreground mb-4 text-base font-medium tracking-wide uppercase">
            You could ask
          </h2>
          <ul className="flex flex-col gap-3">
            <SuggestionCard>What tablets do I take this morning?</SuggestionCard>
            <SuggestionCard>When is my next appointment?</SuggestionCard>
            <SuggestionCard>Am I allergic to anything?</SuggestionCard>
          </ul>
        </div>
      </section>
    </div>
  );
};
