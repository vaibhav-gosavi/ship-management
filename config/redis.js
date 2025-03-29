// config/redis.js
import Redis from 'ioredis';
import logger from '../utils/logger.js'; // Assuming logger is in ../utils
import 'dotenv/config';

// Define Redis connection options based on environment variables
const redisConnectionOptions = {
    host: process.env.REDIS_HOST || '127.0.0.1', // Default to localhost
    port: parseInt(process.env.REDIS_PORT || '6379', 10), // Default Redis port
    // password: process.env.REDIS_PASSWORD || undefined, // Use undefined if no password
    db: parseInt(process.env.REDIS_DB || '0', 10), // Default DB 0
    maxRetriesPerRequest: null, // Recommended setting for BullMQ
    enableReadyCheck: false,   // Recommended setting for BullMQ
    // Optional: Add ioredis specific options if needed
    // connectTimeout: 10000, // e.g., 10s timeout
    // retryStrategy: (times) => Math.min(times * 50, 2000), // e.g., custom retry
};

// Log the configuration being used (mask password)
logger.info('Redis Configuration:', {
    host: redisConnectionOptions.host,
    port: redisConnectionOptions.port,
    password: redisConnectionOptions.password ? '******' : 'Not Set',
    db: redisConnectionOptions.db,
});

// --- IMPORTANT ---
// We export the *options object* itself for BullMQ.
// BullMQ will create and manage its own ioredis connection(s) using these options.
export default redisConnectionOptions;

// --- Optional: Create a separate client instance for monitoring or direct use ---
// If other parts of your app need direct Redis access (e.g., health checks, caching)
// create a separate instance here, but DO NOT export it as the default for BullMQ.

let monitoringClient = null;

try {
    // Attempt to create a client instance *only* for monitoring/direct use
    monitoringClient = new Redis(redisConnectionOptions);

    monitoringClient.on('connect', () => {
        logger.info('Redis monitoring client connected successfully.');
    });

    monitoringClient.on('ready', () => {
         logger.info('Redis monitoring client is ready.');
    });

    monitoringClient.on('error', (err) => {
        // This listener is crucial for catching connection errors for this specific client
        logger.error('Redis monitoring client error:', err.message);
        // Prevent app crash on connection error if this client isn't critical for startup
    });

    monitoringClient.on('close', () => {
        logger.warn('Redis monitoring client connection closed.');
    });

    monitoringClient.on('reconnecting', () => {
         logger.info('Redis monitoring client attempting to reconnect...');
    });

    // You might export a function to access this client if needed elsewhere safely
    // export const getMonitoringRedisClient = () => monitoringClient;

    // Perform a PING test to verify the connection early (optional)
    monitoringClient.ping().then(result => {
         if (result === 'PONG') {
             logger.info('Redis monitoring client PING successful.');
         } else {
              logger.warn('Redis monitoring client PING responded unexpectedly:', result);
         }
    }).catch(err => {
         logger.error('Redis monitoring client PING failed:', err.message);
    });


} catch (error) {
     logger.error('Failed to create Redis monitoring client instance:', error);
     // Application can continue, but direct Redis operations might fail if used
}

// Example of exporting the monitoring client if needed (use with caution)
export { monitoringClient }; // Export named instance for specific use cases