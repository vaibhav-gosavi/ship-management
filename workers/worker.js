// workers/worker.js
import { Worker } from 'bullmq';
import mongoose from 'mongoose'; // Mongoose might be implicitly used by models inside pipelines
import redisConfig from '../config/redis.js'; // Corrected path
import connectDB from '../config/database.js'; 
import logger from '../utils/logger.js'; // Use your structured logger
import { runFuelModelTrainingPipeline } from '../training_pipelines/fuel_pipeline.js';
import { runMaintenanceModelTrainingPipeline } from '../training_pipelines/maintenance_pipeline.js';

// --- Configuration ---
// Allow configuring concurrency via environment variable
const CONCURRENCY = parseInt(process.env.TRAINING_WORKER_CONCURRENCY || '1', 10);

// --- Worker Initialization ---
const startWorker = async () => {
    try {
        logger.info('Connecting Worker to MongoDB...');
        await connectDB() // Ensure DB is connected before starting worker
        logger.info('Worker connected to MongoDB successfully.');

        logger.info(`Initializing Model Training Worker with concurrency ${CONCURRENCY}...`);

        const worker = new Worker(
            'model-training', // Queue name MUST match the one used in the scheduler
            async (job) => {
                // Log job start with details
                logger.info(`Processing job ${job.id} (Type: ${job.name}) - Attempt ${job.attemptsMade + 1}/${job.opts.attempts || 'N/A'}`);
                if (job.data) {
                    logger.debug('Job Data:', job.data); // Log data only in debug mode if sensitive
                }

                let result; // To store the outcome from the pipeline

                try {
                    // Route job to the correct pipeline function
                    switch (job.name) {
                        case 'train_fuel_model':
                            await job.updateProgress(10); // Indicate starting
                            logger.info(`[Job ${job.id}] Starting Fuel Model Training Pipeline...`);
                            result = await runFuelModelTrainingPipeline(job.data);
                            await job.updateProgress(100); // Indicate completion
                            logger.info(`[Job ${job.id}] Fuel model training pipeline finished.`);
                            break;

                        case 'train_maintenance_model':
                            await job.updateProgress(10);
                            logger.info(`[Job ${job.id}] Starting Maintenance Model Training Pipeline...`);
                            result = await runMaintenanceModelTrainingPipeline(job.data);
                            await job.updateProgress(100);
                            logger.info(`[Job ${job.id}] Maintenance model training pipeline finished.`);
                            break;

                        default:
                            logger.warn(`[Job ${job.id}] Unknown job name received: ${job.name}`);
                            // Explicitly throw an error for unknown jobs to fail them
                            throw new Error(`Unknown job name: ${job.name}`);
                    }

                    // Return the specific result from the pipeline function
                    // This gets stored in the completed job details in Redis
                    return result;

                } catch (error) {
                    // Log the specific error during job processing
                    logger.error(`[Job ${job.id}] Error processing job (Type: ${job.name}):`, error);
                    // CRITICAL: Re-throw the error so BullMQ knows the job failed
                    // It will then handle retries based on queue settings.
                    throw error;
                }
            },
            {
                connection: redisConfig, // Use imported Redis config
                concurrency: CONCURRENCY, // Process N jobs concurrently (usually 1 for training)
                // Keep default job options from queue definition unless overriding here
                removeOnComplete: { count: 100 }, // Keep history of completed jobs
                removeOnFail: { count: 500 },      // Keep history of failed jobs
            }
        );

        // --- Worker Event Listeners ---
        worker.on('completed', (job, returnValue) => {
            // Log successful completion, potentially including key parts of the return value
            logger.info(`Worker completed job ${job.id} (Type: ${job.name}) successfully.`);
            if (returnValue) {
                 logger.debug(`[Job ${job.id}] Return Value:`, returnValue);
            }
        });

        worker.on('failed', (job, error) => {
            // Log failed jobs after all retries are exhausted
            logger.error(`Worker failed job ${job.id} (Type: ${job.name}) after ${job.attemptsMade} attempts:`, error);
        });

         worker.on('error', err => {
            // Log errors related to the worker itself (e.g., connection issues)
            logger.error('Model Training Worker encountered an error:', err);
        });

        worker.on('active', (job) => {
             logger.info(`Worker picked up job ${job.id} (Type: ${job.name})`);
        });

         worker.on('progress', (job, progress) => {
            logger.debug(`[Job ${job.id}] Progress updated: ${progress}%`);
        });

        logger.info('Model Training Worker started and listening for jobs...');

        // Graceful shutdown handling (optional but good practice)
        const gracefulShutdown = async () => {
            logger.info('Shutting down model training worker...');
            await worker.close();
            await mongoose.disconnect(); // Close DB connection if worker owns it
            logger.info('Worker shut down gracefully.');
            process.exit(0);
        };
        process.on('SIGTERM', gracefulShutdown); // Signal for termination (e.g., from Docker stop)
        process.on('SIGINT', gracefulShutdown);  // Signal for interrupt (e.g., Ctrl+C)


    } catch (error) {
        logger.error('Failed to start Model Training Worker:', error);
        process.exit(1); // Exit if the worker cannot initialize (e.g., DB connection fails)
    }
};

// ... (previous code remains the same)

if (process.argv[2] === 'maintenance') {
    (async () => {
      await connectDB();
      const result = await runMaintenanceModelTrainingPipeline({
        progressCallback: (progress, message) => {
          console.log(`Progress: ${progress}% - ${message}`);
        }
      });
      console.log('Training results:', result);
      process.exit(0);
    })().catch(err => {
      console.error('Error:', err);
      process.exit(1);
    });
} else if (process.argv[2] === 'fuel') {
    (async () => {
      await connectDB();
      const result = await runFuelModelTrainingPipeline({
        progressCallback: (progress, message) => {
          console.log(`Progress: ${progress}% - ${message}`);
        }
      });
      console.log('Training results:', result);
      process.exit(0);
    })().catch(err => {
      console.error('Error:', err);
      process.exit(1);
    });
} else {
    // Original worker code
    startWorker();
}