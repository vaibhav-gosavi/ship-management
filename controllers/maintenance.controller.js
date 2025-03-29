import Maintenance from '../models/maintenance.model.js';
import Ship from '../models/ship.model.js';
import logger from '../utils/logger.js';
import * as predictionService from '../services/prediction.service.js';

export const addMaintenance = async (req, res) => {
  try {
    const {
      shipId,
      type,
      description,
      performedAt,
      engineHoursAtMaintenance,
      partsReplaced,
      cost,
      technician
    } = req.body;

    // Validate required fields
    if (!shipId || !type || !description || !performedAt || !cost || !technician) {
      return res.status(400).json({ 
        message: 'Please provide all required maintenance details' 
      });
    }

    // Verify ship exists
    const shipExists = await Ship.findById(shipId);
    if (!shipExists) {
      return res.status(404).json({ 
        message: `Ship with ID ${shipId} not found` 
      });
    }

    // Create new maintenance record
    const maintenance = new Maintenance({
      shipId,
      type,
      description,
      performedAt: new Date(performedAt),
      engineHoursAtMaintenance,
      partsReplaced,
      cost,
      technician,
      nextDue: partsReplaced?.[0]?.lifeExpectancy 
        ? new Date(Date.now() + partsReplaced[0].lifeExpectancy * 3600000)
        : undefined
    });

    const savedMaintenance = await maintenance.save();
    res.status(201).json(savedMaintenance);

  } catch (error) {
    console.error('Error adding maintenance:', error);
    res.status(500).json({ 
      message: 'Error recording maintenance activity',
      error: error.message 
    });
  }
};


// {
//     "shipId": "507f1f77bcf86cd799439011",
//     "type": "Engine Overhaul",
//     "description": "Complete engine service and parts replacement",
//     "performedAt": "2023-05-10T09:00:00Z",
//     "engineHoursAtMaintenance": 5000,
//     "partsReplaced": [
//       {
//         "name": "Fuel Injector",
//         "serialNumber": "FI-2023-1234",
//         "lifeExpectancy": 10000
//       },
//       {
//         "name": "Oil Filter",
//         "serialNumber": "OF-2023-5678",
//         "lifeExpectancy": 5000
//       }
//     ],
//     "cost": 8500,
//     "technician": "John Smith"
//  }


export const getMaintenanceSchedule = async (req, res, next) => {
  const { shipId, type } = req.query;

  // --- Validation ---
  if (!shipId) {
      return res.status(400).json({ message: 'Missing required query parameter: shipId' });
  }
  if (!type) {
       return res.status(400).json({ message: 'Missing required query parameter: type (e.g., Engine Check)' });
  }
  
  // Check if the maintenance type is supported (has a trained model)
  // Updated to match your trained model types from the pipeline
  const supportedMaintenanceTypes = ['Engine Overhaul']; // Add other types as you train them
  if (!supportedMaintenanceTypes.includes(type)) {
      return res.status(404).json({ 
          message: `Maintenance prediction for type "${type}" is not available. Supported types: ${supportedMaintenanceTypes.join(', ')}`,
          supportedTypes: supportedMaintenanceTypes
      });
  }

  // Optional: Check if ship exists
  try {
      const shipExists = await Ship.findById(shipId);
      if (!shipExists) {
          return res.status(404).json({ message: `Ship with ID ${shipId} not found.` });
      }
  } catch (error) { 
      console.error('Error checking ship existence:', error);
  }
  // --- End Validation ---

  try {
      const predictionResult = await predictionService.predictNextMaintenanceRUL(shipId, type);

      if (predictionResult.error) {
          // Handle specific prediction errors gracefully
          logger.warn(`Maintenance prediction failed for Ship ${shipId}, Type ${type}: ${predictionResult.error}`);
          
          // Handle model loading error specifically
          if (predictionResult.error.includes('Could not load Maintenance') || 
              predictionResult.error.includes('no such file or directory')) {
              return res.status(404).json({
                  message: `No prediction model available for maintenance type "${type}"`,
                  shipId: shipId,
                  maintenanceType: type,
              });
          }
          
          // Handle other specific errors
          if (predictionResult.error === 'No prior maintenance data' || 
              predictionResult.error === 'Cannot determine current engine hours') {
              return res.status(404).json({
                  message: `Cannot predict RUL: ${predictionResult.error}`,
                  shipId: shipId,
                  maintenanceType: type,
              });
          }
          
          // Generic internal error
          return res.status(500).json({ message: 'Maintenance prediction failed internally.' });
      }

      // Successfully predicted RUL
      res.status(200).json({
          shipId: shipId,
          maintenanceType: type,
          predictedRULHours: predictionResult.predictedRULHours?.toFixed(0), 
          predictedIntervalHours: predictionResult.predictedIntervalHours?.toFixed(0),
          hoursSinceLastMaintenance: predictionResult.hoursSinceLast?.toFixed(0),
          engineHoursAtLastMaintenance: predictionResult.engineHoursAtLastMaintenance,
          currentEngineHoursEstimate: predictionResult.currentEngineHoursEstimate,
          predictionTimestamp: new Date().toISOString(),
          // Add model info if desired (e.g., version, training date)
      });

  } catch (error) {
      logger.error(`Error in getMaintenanceSchedule controller for Ship ${shipId}, Type ${type}:`, error);
      next(error); // Pass to global error handler
  }
};