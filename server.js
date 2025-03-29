// server.js
import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose'; // Added mongoose import for connection state checking
import 'dotenv/config';
import schedule from 'node-schedule';
import swaggerUi from 'swagger-ui-express';
import YAML from 'yamljs';
const swaggerDocument = YAML.load('./docs/api-spec.yaml');

// Import config/index.js for services (Redis, Queues)
import serviceConfig from './config/index.js';
const { redisClient, queues, connectRedis } = serviceConfig; // Destructure needed services
// Make sure 'queues' object exists before destructuring
const modelTrainingQueue = queues?.modelTrainingQueue; // Safely access the queue

// Import connectDB function
import connectDB from './config/database.js';

import predictionService from './services/prediction.service.js';
import routes from './routes/index.js';
import logger from './utils/logger.js';

const app = express();

// --- ASYNCHRONOUS STARTUP FUNCTION ---
const startApp = async () => {
    let serverInstance; // To hold the server instance for graceful shutdown
    try {
        // 1. Connect to MongoDB FIRST
        await connectDB();

        // 2. Connect to Redis
        await connectRedis(); // Ensure Redis connects before scheduler/server start

        // ======================
        // Middleware Setup
        // ======================
        app.use(cors({
             // Configure origins based on environment - '*' is insecure for production
             origin: process.env.CORS_ORIGINS ? process.env.CORS_ORIGINS.split(',') : (process.env.NODE_ENV === 'production' ? false : '*'),
             methods: "GET,HEAD,PUT,PATCH,POST,DELETE",
             credentials: true // Allow cookies if needed for auth
        }));
        app.use(express.json({ limit: '10kb' })); // Limit payload size
        app.use(express.urlencoded({ extended: true })); // Parse URL-encoded bodies

        // HTTP Request Logging Middleware
        app.use((req, _, next) => {
            // Use logger's http level
             logger.http(`${req.method} ${req.originalUrl} - ${req.ip}`);
            next();
        });

        // ======================
        // Route Registration
        // ======================
        app.use('/', routes); // Mount all API routes

        // Add Swagger UI route
        app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

        // ======================
        // Health Checks
        // ======================
        app.get('/health', async (_, res) => { // Make health check async for redis ping
            let redisStatus = 'DISCONNECTED';
            try {
                 if (redisClient?.isOpen) { // Check if client exists and is open
                     const pingResponse = await redisClient.ping();
                     redisStatus = pingResponse === 'PONG' ? 'CONNECTED' : 'UNHEALTHY';
                 } else if (redisClient?.isReady) { // Fallback check for older versions or different states
                      redisStatus = 'CONNECTED (Ready)';
                 }
            } catch (e) {
                logger.warn('Health check Redis ping failed:', e.message);
                 redisStatus = 'ERROR';
            }
            res.status(200).json({
                status: 'UP',
                timestamp: new Date().toISOString(),
                services: {
                    redis: redisStatus,
                    mongo: mongoose.connection.readyState === 1 ? 'CONNECTED' : // 1 === connected
                           mongoose.connection.readyState === 2 ? 'CONNECTING' :
                           mongoose.connection.readyState === 3 ? 'DISCONNECTING' :
                           'DISCONNECTED', // 0 or other states
                    queues: queues ? Object.keys(queues) : 'Not Initialized'
                }
            });
        });

        // ==========================
        // Setup Model Training Scheduler
        // ==========================
        let trainingJobSchedule = 'DISABLED'; // Default status
        if (modelTrainingQueue) {
            trainingJobSchedule = process.env.TRAINING_JOB_SCHEDULE || '0 2 * * 0'; // Default: 2 AM Sunday
            logger.info(`Scheduling model training job with pattern: ${trainingJobSchedule}`);

            schedule.scheduleJob(trainingJobSchedule, async () => {
                logger.info(`Scheduler triggered [${trainingJobSchedule}]: Adding training jobs...`);
                try {
                    // 1. Clear prediction model cache
                    logger.info('Clearing model cache before adding training jobs...');
                    predictionService.clearModelCache();

                    // 2. Add Fuel Model Training Job
                    await modelTrainingQueue.add('train_fuel_model', {
                        triggeredBy: 'scheduler',
                        timestamp: new Date().toISOString()
                    }, { jobId: `fuel-train-sched-${Date.now()}` });
                    logger.info('Added scheduled train_fuel_model job to the queue.');

                    // 3. Add Maintenance Model Training Job
                    await modelTrainingQueue.add('train_maintenance_model', {
                        triggeredBy: 'scheduler',
                        timestamp: new Date().toISOString()
                    }, { jobId: `maint-train-sched-${Date.now()}` });
                    logger.info('Added scheduled train_maintenance_model job to the queue.');

                    logger.info('Scheduled training jobs successfully added.');
                } catch (error) {
                    logger.error('Error scheduling weekly training jobs:', error);
                }
            });
            logger.info('Weekly model training scheduler initialized.');
        } else {
            logger.warn('Model Training Queue not available. Weekly training jobs will NOT be scheduled.');
        }

        // ======================
        // Error Handling
        // ======================
        // 404 Handler - Catch all for routes not defined above
        app.use((req, res) => {
            logger.warn(`404 Not Found - ${req.method} ${req.originalUrl}`);
            res.status(404).json({ error: 'Not Found', path: req.originalUrl });
        });

        // Global error handler - Must have 4 arguments for Express to recognize it as error handler
        app.use((err, req, res, next) => {
            // Log the error fully
            logger.error(`Server Error on ${req.method} ${req.originalUrl}: ${err.message}`, {
                stack: err.stack, // Include stack trace
                statusCode: err.statusCode, // Custom status code if available
                isOperational: err.isOperational, // Custom flag for expected errors
                requestBody: req.body, // Be cautious logging request bodies if they contain sensitive data
                requestQuery: req.query
            });

            // Determine status code
            const statusCode = err.statusCode || 500;

            // Send appropriate response based on environment and error type
            res.status(statusCode).json({
                status: 'error',
                message: (process.env.NODE_ENV === 'production' && !err.isOperational && statusCode === 500)
                    ? 'Internal Server Error' // Generic message for unexpected production errors
                    : err.message || 'An unexpected error occurred', // Specific message otherwise
                // Optionally include stack in development
                ...(process.env.NODE_ENV !== 'production' && { stack: err.stack })
            });
        });

        // ======================
        // Server Startup
        // ======================
        const PORT = process.env.PORT || 5001;
        serverInstance = app.listen(PORT, () => { // Assign to serverInstance
            logger.info(`
    🚢 Ship Management API Running
    =============================
    Mode: ${process.env.NODE_ENV || 'development'}
    Port: ${PORT}
    Redis: ${redisClient?.isOpen ? '✅' : '❌'}
    MongoDB: ${mongoose.connection.readyState === 1 ? '✅' : '❌'}
    Queues: ${queues ? Object.keys(queues).join(', ') : '❌'}
    Training Schedule: ${trainingJobSchedule}
    `);
        });

    } catch (error) {
        logger.error('❌ Failed to start application:', error);
        process.exit(1);
    }


    // --- Graceful Shutdown Logic ---
    const shutdown = async (signal) => {
        logger.warn(`Received signal: ${signal}. Shutting down gracefully...`);

        // Stop accepting new connections
        if (serverInstance) {
             serverInstance.close(async () => {
                logger.info('HTTP server closed.');

                // Shutdown scheduler
                logger.info('Shutting down scheduled jobs...');
                await schedule.gracefulShutdown();
                logger.info('Scheduler shut down.');

                // Close Redis connection
                if (redisClient?.isOpen) {
                    try {
                         await redisClient.quit();
                         logger.info('Redis connection closed.');
                    } catch(redisErr) {
                         logger.error('Error closing Redis connection:', redisErr);
                    }
                }

                // Close MongoDB connection
                try {
                    await mongoose.disconnect();
                    logger.info('MongoDB connection closed.');
                } catch(mongoErr) {
                     logger.error('Error closing MongoDB connection:', mongoErr);
                }

                logger.info('Application shut down complete.');
                process.exit(0); // Exit cleanly
            });

             // Force shutdown after timeout if connections don't close
             setTimeout(() => {
                 logger.error('Could not close connections in time, forcing shutdown.');
                 process.exit(1);
             }, 10000); // 10 second timeout
        } else {
             // If server never started, just exit
             process.exit(0);
        }
    };

    process.on('SIGTERM', () => shutdown('SIGTERM')); // Standard signal for termination
    process.on('SIGINT', () => shutdown('SIGINT'));   // Signal for interrupt (Ctrl+C)

}; // --- End of startApp Function ---


// --- Start the Application ---
startApp();