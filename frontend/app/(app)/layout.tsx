import { headers } from 'next/headers';
import { auth, signOut } from '@/auth';
import { getAppConfig } from '@/lib/utils';
import { SessionErrorHandler } from '@/components/session-error-handler';

interface LayoutProps {
  children: React.ReactNode;
}

export default async function Layout({ children }: LayoutProps) {
  const hdrs = await headers();
  const { companyName, logo, logoDark } = await getAppConfig(hdrs);
  const session = await auth();

  return (
    <>
      <SessionErrorHandler />
      <header className="fixed top-0 left-0 z-50 hidden w-full flex-row justify-between p-6 md:flex">
        <div />
        <div className="flex items-center gap-4">
          {session?.user && (
            <span className="text-foreground font-mono text-xs">
              {session.user.name ?? session.user.email}
            </span>
          )}
          <a href="/history" className="text-foreground font-mono text-xs underline underline-offset-4">
            会話履歴
          </a>
          <a href="/rag" className="text-foreground font-mono text-xs underline underline-offset-4">
            RAG管理
          </a>
          {session?.user && (
            <form
              action={async () => {
                'use server';
                await signOut({ redirectTo: '/' });
              }}
            >
              <button
                type="submit"
                className="text-foreground font-mono text-xs underline underline-offset-4"
              >
                ログアウト
              </button>
            </form>
          )}
        </div>
      </header>
      {children}
    </>
  );
}
