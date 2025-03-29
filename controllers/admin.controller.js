// controllers/admin.controller.js (Example)
import { modelTrainingQueue } from '../config/queue.js';
import predictionService from '../services/prediction.service.js';
import logger from '../utils/logger.js';

export const triggerTraining = async (req, res) => {
    try {
        logger.info('Manual training trigger requested.');
        predictionService.clearModelCache(); // Clear cache first
        await modelTrainingQueue.add('train_fuel_model', { triggeredBy: 'manual_api' });
        await modelTrainingQueue.add('train_maintenance_model', { triggeredBy: 'manual_api' });
        logger.info('Manual training jobs added to queue.');
        res.status(202).json({ message: 'Training jobs added to the queue.' });
    } catch (error) {
        logger.error('Error triggering manual training:', error);
        res.status(500).json({ message: 'Failed to add training jobs.' });
    }
};