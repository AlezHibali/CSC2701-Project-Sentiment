import { NextRequest, NextResponse } from 'next/server';
import { classifySentiment } from '@/lib/sagemaker';

const ENDPOINT_NAME = process.env.SAGEMAKER_ENDPOINT_NAME || 'twitter-roberta-sentiment-serverless';

export async function POST(request: NextRequest) {
    try {
        const { tweet } = await request.json();

        if (!tweet || typeof tweet !== 'string') {
            return NextResponse.json({ error: 'Invalid tweet input' }, { status: 400 });
        }

        if (!ENDPOINT_NAME) {
            return NextResponse.json({ error: 'SageMaker endpoint not configured' }, { status: 500 });
        }

        const { label, score } = await classifySentiment(tweet, ENDPOINT_NAME);

        return NextResponse.json({ label, score });
    } catch (error) {
        console.error('API error:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}