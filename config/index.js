// config/index.js (Revised - ONLY handles Redis & Queues)
import { Queue } from 'bullmq'; // *** SWITCHED TO BullMQ ***
import { createClient } from 'redis';
import 'dotenv/config';
import logger from '../utils/logger.js';

// --- Redis Setup ---
// Define Redis options based on URL or specific host/port
const redisOptions = {
    // Prefer specific options if available, fallback to URL
    host: process.env.REDIS_HOST || '127.0.0.1', // Changed from localhost for clarity
    port: parseInt(process.env.REDIS_PORT || '6379', 10),
    password: process.env.REDIS_PASSWORD || undefined, // Use undefined if not set
    // Alternatively, use url if that's preferred:
    // url: process.env.REDIS_URL || 'redis://127.0.0.1:6379'
};

const redisClient = createClient({
    // Use password directly if provided, otherwise use the options object
    password: redisOptions.password,
    socket: {
        host: redisOptions.host,
        port: redisOptions.port
    }
    // Or using url:
    // url: redisOptions.url
});

redisClient.on('error', (err) => logger.error('Redis Client Error:', err));
redisClient.on('connect', () => logger.info('Connecting to Redis...'));
redisClient.on('ready', () => logger.info('Redis client ready.'));
redisClient.on('end', () => logger.warn('Redis client connection closed.'));

// Connect Redis Client - Needs to be connected *before* BullMQ uses it
// It's better to handle this connection explicitly before initializing queues
const connectRedis = async () => {
    try {
        if (!redisClient.isOpen) {
             await redisClient.connect();
        }
    } catch (err) {
        logger.error('Failed to connect Redis Client:', err);
        // Decide if you want to exit or continue without queues potentially
         process.exit(1); // Exit if Redis is critical
    }
};


// --- BullMQ Queue Setup ---
// Initialize queues *after* ensuring Redis is trying to connect
// BullMQ uses the redisOptions directly for its own connection management
const queues = {
    modelTrainingQueue: new Queue('model-training', {
        connection: redisOptions, // Pass the options object
        // ... default job options ...
    }),
    fuelPredictionQueue: new Queue('fuel-prediction', {
        connection: redisOptions,
        // ... default job options ...
    }),
    routeOptimizationQueue: new Queue('route-optimization', {
        connection: redisOptions,
        // ... default job options ...
    })
    // Add other queues as needed (maintenanceQueue, routeQueue, fuelQueue from original)
};

// Setup listeners (as shown in the previous queue.js example)
// setupQueueListeners('model-training', queues.modelTrainingQueue);
// ... etc. for other queues

// --- Export initialized services ---
export default {
  redisClient,      // Export the client instance
  redisOptions,     // Export options if needed elsewhere
  queues,           // Export the queue instances
  connectRedis      // Export the function to connect Redis explicitly
};