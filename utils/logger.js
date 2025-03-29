// utils/logger.js
import winston from 'winston';
import path from 'path';
import fs from 'fs';

// Ensure log directory exists (important for file transport)
const logDir = path.resolve(process.cwd(), 'logs');
if (!fs.existsSync(logDir)) {
    try {
        fs.mkdirSync(logDir);
    } catch (error) {
        console.error('Could not create log directory:', error);
        // Proceed without file logging if directory creation fails
    }
}

// Define custom log levels if needed (Winston's defaults are usually fine)
const levels = winston.config.npm.levels; // error: 0, warn: 1, info: 2, http: 3, verbose: 4, debug: 5, silly: 6

// Determine log level based on environment variable or NODE_ENV
const level = () => {
    const envLogLevel = process.env.LOG_LEVEL?.toLowerCase();
    if (envLogLevel && levels[envLogLevel] !== undefined) {
        return envLogLevel;
    }
    return process.env.NODE_ENV === 'production' ? 'warn' : 'debug';
};

// Define different formats for development and production
const colorizer = winston.format.colorize();

// Development format: Timestamp, Colorized Level, Message, Stack Trace (if error)
const devFormat = winston.format.combine(
    winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
    winston.format.errors({ stack: true }), // Include stack trace in the 'error' object
    winston.format.printf(({ timestamp, level, message, stack }) => {
        const colorizedLevel = colorizer.colorize(winston.level, level); // Manually colorize
        // If stack exists (from an Error object), print it, otherwise just the message
        const logMessage = stack ? `${stack}` : message;
        return `${timestamp} ${colorizedLevel}: ${logMessage}`;
    })
);

// Production format: Timestamp, Level, Message, Stack Trace (if error) - as JSON
const prodFormat = winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }), // Ensure stack trace is part of the log object
    winston.format.json() // Output log entry as JSON
);

// Choose the format based on the environment
const format = process.env.NODE_ENV === 'production' ? prodFormat : devFormat;

// Define transports (where the logs go)
const transports = [
    // Always log to the console
    new winston.transports.Console({
        // Use the determined level for console in dev, but maybe 'info' or higher in prod console
        level: process.env.NODE_ENV === 'production' ? 'info' : level(),
        // Use appropriate format based on environment
        format: format,
        // Handle exceptions and rejections that weren't caught elsewhere
        handleExceptions: true,
        handleRejections: true,
    }),
];

// Add file transports ONLY in production environment
if (process.env.NODE_ENV === 'production' && fs.existsSync(logDir)) {
    transports.push(
        new winston.transports.File({
            filename: path.join(logDir, 'error.log'),
            level: 'error', // Log only errors and above to this file
            format: prodFormat, // Always use JSON format for files
            maxsize: 5242880, // 5MB
            maxFiles: 5, // Keep up to 5 rotated files
            handleExceptions: true, // Also log uncaught exceptions here
            handleRejections: true,
        })
    );
    transports.push(
        new winston.transports.File({
            filename: path.join(logDir, 'combined.log'),
            level: level(), // Log according to the determined level (e.g., 'warn' or 'info')
            format: prodFormat, // Always use JSON format for files
            maxsize: 10485760, // 10MB
            maxFiles: 5,
            handleExceptions: true,
            handleRejections: true,
        })
    );
}

// Create the logger instance
const logger = winston.createLogger({
    level: level(), // Set the maximum level to log
    levels: levels,
    format: format, // Master format (applied if transport doesn't have its own)
    transports: transports,
    exitOnError: false, // Do not exit application on handled exceptions logged by Winston
});

// Add a stream method for integration with HTTP logging middleware like Morgan (optional)
logger.stream = {
    write: (message) => {
        // Use the 'http' level if defined, otherwise 'info'
        logger.http(message.trim());
    },
};

// Log confirmation
logger.info(`Logger initialized with level: ${level()}`);
if (process.env.NODE_ENV === 'production') {
    logger.info(`Production logging enabled (Console level: info, File level: ${level()})`);
} else {
    logger.info('Development logging enabled (Console colorized)');
}


// Export the configured logger
export default logger;