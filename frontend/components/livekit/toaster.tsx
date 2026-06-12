'use client';
import { useTheme } from 'next-themes';
import { Toaster as Sonner, ToasterProps } from 'sonner';

export function Toaster({ ...props }: ToasterProps) {
  const { theme = 'system' } = useTheme();
  return (
    <>
      <style>{`
        [data-sonner-toaster] {
          top: 1rem !important;
          left: 1rem !important;
          bottom: auto !important;
          right: auto !important;
          width: auto !important;
          max-width: 364px !important;
        }
        @media (max-width: 600px) {
          [data-sonner-toaster] {
            top: 1rem !important;
            left: 1rem !important;
            bottom: auto !important;
            right: auto !important;
            width: auto !important;
            max-width: calc(100vw - 2rem) !important;
          }
          [data-sonner-toaster] [data-sonner-toast] {
            left: 0 !important;
            right: auto !important;
            width: 100% !important;
          }
        }
      `}</style>
      <Sonner
        theme={theme as ToasterProps['theme']}
        className="toaster group"
        position="top-left"
        style={
          {
            '--normal-bg': 'var(--popover)',
            '--normal-text': 'var(--popover-foreground)',
            '--normal-border': 'var(--border)',
          } as React.CSSProperties
        }
        {...props}
      />
    </>
  );
}
