import { SageMakerRuntimeClient, InvokeEndpointCommand } from '@aws-sdk/client-sagemaker-runtime';

const client = new SageMakerRuntimeClient({
    region: process.env.AWS_REGION || 'us-east-1',
});

export async function classifySentiment(tweet: string, endpointName: string): Promise<{ label: string; score: number }> {
    const payload = JSON.stringify({ text: tweet });

    const command = new InvokeEndpointCommand({
        EndpointName: endpointName,
        ContentType: 'application/json',
        Accept: 'application/json',
        Body: new TextEncoder().encode(payload),
    });

    try {
        const response = await client.send(command);
        const result = new TextDecoder().decode(response.Body);
        const parsed = JSON.parse(result);

        return parsed;
    } catch (error) {
        console.error('SageMaker invocation error:', error);
        throw new Error('Failed to classify sentiment');
    }
}