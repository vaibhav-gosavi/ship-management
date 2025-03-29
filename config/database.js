// config/database.js
import mongoose from 'mongoose';
import 'dotenv/config';
import logger from '../utils/logger.js'; // Use your logger

const connectDB = async () => {
  try {
    // Remove deprecated options
    await mongoose.connect(process.env.MONGO_URI);

    const conn = mongoose.connection; // Get connection details
    logger.info(`MongoDB Connected: ${conn.host} - Database: ${conn.name}`);

    // Optional: Add listeners for disconnection events if needed
    conn.on('error', (err) => logger.error('MongoDB connection error:', err));
    conn.on('disconnected', () => logger.warn('MongoDB disconnected.'));

  } catch (err) {
    logger.error('MongoDB initial connection error:', err.message);
    // Exit process on initial connection failure - critical dependency
    process.exit(1);
  }
};

// Export only the function
export default connectDB;