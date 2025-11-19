'use client';

import { useState } from 'react';

interface ClassificationResult {
    label: string;
    score: number;
}

export default function Home() {
    const [tweet, setTweet] = useState('');
    const [sentiment, setSentiment] = useState<string | null>(null);
    const [score, setScore] = useState<number | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleClassify = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!tweet.trim()) return;

        setSentiment(null);
        setScore(null);
        setError(null);

        setLoading(true);

        try {
            const response = await fetch('/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tweet }),
            });

            if (!response.ok) {
                throw new Error('Classification failed');
            }

            const data: ClassificationResult = await response.json();

            setSentiment(data.label);
            setScore(data.score);

        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    const getSentimentClasses = (sentiment: string) => {
        switch (sentiment) {
            case 'positive':
                return 'bg-green-100 border-green-300 text-green-800';
            case 'neutral':
                return 'bg-amber-100 border-amber-300 text-amber-800';
            case 'negative':
                return 'bg-red-100 border-red-300 text-red-800';
            default:
                return 'bg-gray-100 border-gray-300 text-gray-800';
        }
    };

    return (
        <main className="flex min-h-screen flex-col items-center justify-center p-24">
            <div className="max-w-md w-full space-y-8">
                <div>
                    <h1 className="text-3xl font-bold text-center text-gray-900">
                        Twitter Sentiment Classifier
                    </h1>
                    <p className="mt-2 text-sm text-gray-500 text-center">
                        Enter a tweet to classify its sentiment (positive, negative, neutral).
                    </p>
                </div>

                <form onSubmit={handleClassify} className="space-y-4">
                    <div className="mb-4">
                        <label htmlFor="tweet" className="block text-sm font-medium text-gray-700">
                            Tweet Text
                        </label>
                        <textarea
                            id="tweet"
                            value={tweet}
                            onChange={(e) => setTweet(e.target.value)}
                            rows={4}
                            className="mt-2 block w-full rounded-md border border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 p-2"
                            placeholder="Enter a tweet..."
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading || !tweet.trim()}
                        className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 cursor-pointer transition duration-150 ease-in-out"
                    >
                        {loading ? 'Classifying...' : 'Classify Sentiment'}
                    </button>
                </form>

                <div className="mt-4 min-h-24">
                    {error && (
                        <div className="text-red-600 text-sm text-center p-3 bg-red-50 border border-red-200 rounded-md">
                            {error}
                        </div>
                    )}

                    {sentiment && (
                        <div className={`text-center p-4 border rounded-md ${getSentimentClasses(sentiment)}`}>
                            <h2 className="text-xl font-bold capitalize">
                                Sentiment: {sentiment}
                            </h2>
                            {score !== null && (
                                <p className="mt-2 text-sm">
                                    Confidence Score: <span className="font-semibold">{score.toFixed(4)}</span>
                                </p>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </main>
    );
}