import { NextResponse } from 'next/server';
import { AccessToken, type AccessTokenOptions, type VideoGrant } from 'livekit-server-sdk';
import { RoomConfiguration } from '@livekit/protocol';
import { auth } from '@/auth';

type ConnectionDetails = {
  serverUrl: string;
  roomName: string;
  participantName: string;
  participantToken: string;
};

const API_KEY = process.env.LIVEKIT_API_KEY;
const API_SECRET = process.env.LIVEKIT_API_SECRET;
const LIVEKIT_URL = process.env.NEXT_PUBLIC_LIVEKIT_URL ?? process.env.LIVEKIT_URL;

export const revalidate = 0;

export async function POST(req: Request) {
  try {
    if (LIVEKIT_URL === undefined) {
      throw new Error('NEXT_PUBLIC_LIVEKIT_URL or LIVEKIT_URL is not defined');
    }
    if (API_KEY === undefined) {
      throw new Error('LIVEKIT_API_KEY is not defined');
    }
    if (API_SECRET === undefined) {
      throw new Error('LIVEKIT_API_SECRET is not defined');
    }

    // セッションからentityIdを取得
    const session = await auth();
    const entityId = session?.user?.id ?? 'anonymous';

    const body = await req.json();
    const agentName: string = body?.room_config?.agents?.[0]?.agent_name;

    const participantName = session?.user?.name ?? 'user';
    const participantIdentity = `voice_assistant_user_${Math.floor(Math.random() * 10_000)}`;
    const roomName = `voice_assistant_room_${Math.floor(Math.random() * 10_000)}`;

    const participantToken = await createParticipantToken(
      {
        identity: participantIdentity,
        name: participantName,
        // entityIdをmetadataに仕込む → agent.pyで受け取る（Phase 2）
        metadata: JSON.stringify({ entityId }),
      },
      roomName,
      agentName
    );

    const data: ConnectionDetails = {
      serverUrl: LIVEKIT_URL,
      roomName,
      participantToken: participantToken,
      participantName,
    };

    return NextResponse.json(data, {
      headers: new Headers({ 'Cache-Control': 'no-store' }),
    });
  } catch (error) {
    if (error instanceof Error) {
      console.error(error);
      return new NextResponse(error.message, { status: 500 });
    }
  }
}

function createParticipantToken(
  userInfo: AccessTokenOptions,
  roomName: string,
  agentName?: string
): Promise<string> {
  const at = new AccessToken(API_KEY, API_SECRET, {
    ...userInfo,
    ttl: '15m',
  });
  const grant: VideoGrant = {
    room: roomName,
    roomJoin: true,
    canPublish: true,
    canPublishData: true,
    canSubscribe: true,
  };
  at.addGrant(grant);
  if (agentName) {
    at.roomConfig = new RoomConfiguration({
      agents: [{ agentName }],
    });
  }
  return at.toJwt();
}
