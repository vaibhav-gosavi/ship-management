// config/queue.js
import { Queue } from 'bullmq'; // Changed import
import redisConfig from './redis.js'; // Assuming this exports connection options { host, port, password?, db? }
import logger from '../utils/logger.js'; // Import logger for queue events

// Helper function to attach common listeners (like error handling)
const setupQueueListeners = (queueName, queueInstance) => {
    queueInstance.on('error', err => {
        // Log Redis connection errors or other queue-level issues
        logger.error(`Queue [${queueName}] Error:`, err);
    });
     // Add other general queue listeners if needed (e.g., 'waiting', but can be verbose)
     logger.info(`Initialized Queue: ${queueName}`);
};

// --- Define Queues ---

// 1. Model Training Queue (Needed for our pipelines)
export const modelTrainingQueue = new Queue('model-training', {
    connection: redisConfig, // Use connection object
    defaultJobOptions: {
        attempts: 2, // Default attempts for training jobs
        backoff: {
            type: 'exponential',
            delay: 5 * 60 * 1000, // 5 minutes backoff
        },
        removeOnComplete: { count: 100 }, // Keep history
        removeOnFail: { count: 500 },
    },
});
setupQueueListeners('model-training', modelTrainingQueue);


// --- Your existing queues (converted to BullMQ) ---

// 2. Fuel Prediction Queue (If used elsewhere, keep; otherwise, maybe remove if only training is backgrounded)
// Note: Real-time prediction usually happens in the API request, not a queue.
// This queue might be intended for something else? Let's keep it for now.
export const fuelPredictionQueue = new Queue('fuel-prediction', {
    connection: redisConfig, // Use connection object
    defaultJobOptions: {
        attempts: 3,
        backoff: {
            type: 'exponential',
            delay: 1000 // 1 second backoff
        }
    }
});
setupQueueListeners('fuel-prediction', fuelPredictionQueue);

// 3. Route Optimization Queue (If used for complex route planning)
export const routeOptimizationQueue = new Queue('route-optimization', {
    connection: redisConfig, // Use connection object
    // BullMQ limiter syntax is slightly different if needed, but the concept exists.
    // Let's keep the options simple for now unless specific rate limiting is required *here*.
    defaultJobOptions: {
         attempts: 1 // Optimization might not be retryable easily
    }
    // If you need BullMQ rate limiting: https://docs.bullmq.io/guide/rate-limiting
});
setupQueueListeners('route-optimization', routeOptimizationQueue);


// --- Remove Old Bull Listeners ---
// Job lifecycle events like 'completed' and 'failed' are best handled
// by the *Worker* processing the jobs, not globally on the Queue object.
// Removing these:
// fuelPredictionQueue.on('completed', ...)
// fuelPredictionQueue.on('failed', ...)


// --- Exports ---
// Export all queues individually using named exports
// This allows importing only the specific queue needed, e.g.,
// import { modelTrainingQueue } from './config/queue.js'; in server.js or worker.js

// The default export is optional if you primarily use named imports
export default {
  modelTrainingQueue, // Add the new queue here too
  fuelPredictionQueue,
  routeOptimizationQueue,
};

logger.info('BullMQ queues initialized.');