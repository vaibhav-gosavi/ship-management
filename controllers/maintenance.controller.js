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


// export const getMaintenanceSchedule = async (req, res, next) => {
//   const { shipId, type } = req.query;

//   // --- Validation ---
//   if (!shipId) {
//       return res.status(400).json({ message: 'Missing required query parameter: shipId' });
//   }
//   if (!type) {
//        return res.status(400).json({ message: 'Missing required query parameter: type (e.g., Engine Check)' });
//   }
  
//   // Check if the maintenance type is supported (has a trained model)
//   // Updated to match your trained model types from the pipeline
//   const supportedMaintenanceTypes = ['Engine Overhaul']; // Add other types as you train them
//   if (!supportedMaintenanceTypes.includes(type)) {
//       return res.status(404).json({ 
//           message: `Maintenance prediction for type "${type}" is not available. Supported types: ${supportedMaintenanceTypes.join(', ')}`,
//           supportedTypes: supportedMaintenanceTypes
//       });
//   }

//   // Optional: Check if ship exists
//   try {
//       const shipExists = await Ship.findById(shipId);
//       if (!shipExists) {
//           return res.status(404).json({ message: `Ship with ID ${shipId} not found.` });
//       }
//   } catch (error) { 
//       console.error('Error checking ship existence:', error);
//   }
//   // --- End Validation ---

//   try {
//       const predictionResult = await predictionService.predictNextMaintenanceRUL(shipId, type);

//       if (predictionResult.error) {
//           // Handle specific prediction errors gracefully
//           logger.warn(`Maintenance prediction failed for Ship ${shipId}, Type ${type}: ${predictionResult.error}`);
          
//           // Handle model loading error specifically
//           if (predictionResult.error.includes('Could not load Maintenance') || 
//               predictionResult.error.includes('no such file or directory')) {
//               return res.status(404).json({
//                   message: `No prediction model available for maintenance type "${type}"`,
//                   shipId: shipId,
//                   maintenanceType: type,
//               });
//           }
          
//           // Handle other specific errors
//           if (predictionResult.error === 'No prior maintenance data' || 
//               predictionResult.error === 'Cannot determine current engine hours') {
//               return res.status(404).json({
//                   message: `Cannot predict RUL: ${predictionResult.error}`,
//                   shipId: shipId,
//                   maintenanceType: type,
//               });
//           }
          
//           // Generic internal error
//           return res.status(500).json({ message: 'Maintenance prediction failed internally.' });
//       }

//       // Successfully predicted RUL
//       res.status(200).json({
//           shipId: shipId,
//           maintenanceType: type,
//           predictedRULHours: predictionResult.predictedRULHours?.toFixed(0), 
//           predictedIntervalHours: predictionResult.predictedIntervalHours?.toFixed(0),
//           hoursSinceLastMaintenance: predictionResult.hoursSinceLast?.toFixed(0),
//           engineHoursAtLastMaintenance: predictionResult.engineHoursAtLastMaintenance,
//           currentEngineHoursEstimate: predictionResult.currentEngineHoursEstimate,
//           predictionTimestamp: new Date().toISOString(),
//           // Add model info if desired (e.g., version, training date)
//       });

//   } catch (error) {
//       logger.error(`Error in getMaintenanceSchedule controller for Ship ${shipId}, Type ${type}:`, error);
//       next(error); // Pass to global error handler
//   }
// };


import mongoose from 'mongoose';
// import Maintenance from '../models/maintenance.model.js';
// import Ship from '../models/ship.model.js';
// import logger from '../utils/logger.js';
// import * as predictionService from '../services/prediction.service.js';
import  TelemetryService from '../services/telemetry.service.js';
import fetch from 'node-fetch';


// Maintenance Agent Class
class MaintenanceAgent {
  constructor(shipId, maintenanceType) {
    this.shipId = shipId;
    this.type = maintenanceType;
    this.decisionTree = {
      primaryStrategy: 'ml_prediction',
      fallbacks: [
        'historical_analysis',
        'expert_system',
        'real_time_monitoring',
        'llm_fallback'
      ]
    };
    this.context = {
      ship: null,
      maintenanceHistory: null,
      telemetry: null
    };
    this.isSupportedType = ['Engine Overhaul'].includes(maintenanceType);
  }
  

  attemptRecovery(error) {
    logger.error(`Recovery attempt for: ${error.message}`);
    return 'Reset context and enforced fallback strategies';
  }

  calculateMLConfidence(prediction) {
    const hoursSince = prediction.hoursSinceLast;
    const interval = prediction.predictedIntervalHours;
    return hoursSince && interval 
      ? Math.min(0.9, 1 - (Math.abs(hoursSince - interval) / interval))
      : 0.5;
  }

  calculateHistoricalIntervals() {
    const history = this.context.maintenanceHistory;
    if (!history || history.length < 2) return { average: 720 };
    
    const intervals = [];
    for (let i = 1; i < history.length; i++) {
      intervals.push((history[i-1].performedAt - history[i].performedAt) / 3600000);
    }
    return {
      average: Math.max(24, intervals.reduce((a,b) => a + b, 0) / intervals.length),
      min: Math.min(...intervals),
      max: Math.max(...intervals)
    };
  }

  getBusinessRules() {
    return {
      'Engine Overhaul': { standardInterval: 5000, criticality: 'high' },
      'Routine Service': { standardInterval: 720, criticality: 'medium' },
      'Emergency Repair': { standardInterval: 0, criticality: 'critical' },
      'Preventive Maintenance': { standardInterval: 1440, criticality: 'low' }
    };
  }

  analyzeTelemetry() {
    const telemetry = this.context.telemetry;
    if (!telemetry) return 'unknown';
    
    const warningThresholds = {
      temperature: 90,
      vibration: 7.5,
      pressure: 110
    };
    
    const warnings = Object.keys(warningThresholds)
      .filter(k => telemetry[k] > warningThresholds[k]);
    
    return warnings.length > 2 ? 'poor' :
           warnings.length > 0 ? 'fair' : 'good';
  }

  async buildContext() {
    try {
      const [ship, history, telemetry] = await Promise.all([
        Ship.findById(this.shipId),
        Maintenance.find({ 
          shipId: this.shipId, 
          type: this.type 
        }).sort({ performedAt: -1 }).limit(10),
        TelemetryService.getLatest(this.shipId)
      ]);

      this.context = {
        ship,
        maintenanceHistory: history,
        telemetry,
        hasMLModel: ['Engine Overhaul'].includes(this.type)
      };
    } catch (error) {
      logger.error('Context build failed:', error);
      throw new Error('Failed to build execution context');
    }
  }

  async executeStrategy(strategy) {
    switch (strategy) {
      case 'ml_prediction':
        if (!this.context.hasMLModel) return { success: false };
        return this.mlStrategy();
      case 'historical_analysis':
        return this.historicalStrategy();
      case 'expert_system':
        return this.expertSystemStrategy();
      case 'real_time_monitoring':
        return this.realTimeStrategy();
      case 'llm_fallback':
        return this.llmStrategy();
      default:
        return { success: false };
    }
  }

  async mlStrategy() {
    try {
      const prediction = await predictionService.predictNextMaintenanceRUL(
        this.shipId, 
        this.type
      );
      
      return {
        strategy: 'ml_prediction',
        success: true,
        data: {
          predictedRULHours: prediction.predictedRULHours,
          predictedIntervalHours: prediction.predictedIntervalHours,
          confidence: this.calculateMLConfidence(prediction)
        }
      };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async historicalStrategy() {
    if (!this.context.maintenanceHistory?.length) {
      return { success: false };
    }
    
    const intervals = this.calculateHistoricalIntervals();
    return {
      strategy: 'historical_analysis',
      success: true,
      data: {
        averageInterval: intervals.average,
        lastMaintenanceDate: this.context.maintenanceHistory[0].performedAt,
        recommendedNext: new Date(
          this.context.maintenanceHistory[0].performedAt.getTime() + 
          intervals.average * 3600000)
      }
    };
  }

  async expertSystemStrategy() {
    const rules = this.getBusinessRules();
    return {
      strategy: 'expert_system',
      success: true,
      data: {
        recommendedInterval: rules[this.type]?.standardInterval || 720,
        criticality: rules[this.type]?.criticality || 'medium'
      }
    };
  }

  async realTimeStrategy() {
    if (!this.context.telemetry) return { success: false };
    
    return {
      strategy: 'real_time_monitoring',
      success: true,
      data: {
        currentHealth: this.analyzeTelemetry(),
        recommendation: 'Schedule diagnostic inspection'
      }
    };
  }

  async llmStrategy() {
    const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
    if (!GEMINI_API_KEY) return { success: false, error: 'No LLM API key' };

    try {
      const prompt = `As a marine maintenance expert analyzing:
      - Ship: ${this.context.ship?.name || 'Unknown'}
      - Maintenance Type: ${this.type}
      - Last Service: ${this.context.maintenanceHistory?.[0]?.performedAt || 'Never'}
      - Current Health: ${this.analyzeTelemetry()}

      Provide:
      1. Recommended interval in hours
      2. Confidence level (High/Medium/Low)
      3. Key factors
      4. Any warnings`;

      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }]
          })
        }
      );

      const data = await response.json();
      const analysis = data.candidates[0].content.parts[0].text;

      return {
        strategy: 'llm_fallback',
        success: true,
        data: {
          analysis,
          confidence: 'medium',
          source: 'gemini_ai'
        }
      };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  formatResponse(result) {
    if (!result.success) {
      return {
        status: 404,
        body: {
          message: `Could not determine maintenance schedule for ${this.type}`,
          availableStrategies: this.decisionTree.fallbacks
        }
      };
    }
    
    const baseResponse = {
      shipId: this.shipId,
      maintenanceType: this.type,
      predictionSource: result.strategy,
      timestamp: new Date().toISOString()
    };
    
    return {
      status: 200,
      body: { ...baseResponse, ...result.data }
    };
  }

  async execute() {
    try {
      await this.buildContext();
      
      let result;
      
      // For unsupported types, skip straight to LLM fallback
      if (!this.isSupportedType) {
        result = await this.llmStrategy();
      } else {
        // For supported types, try all strategies in order
        const strategies = [this.decisionTree.primaryStrategy, ...this.decisionTree.fallbacks];
        for (const strategy of strategies) {
          result = await this.executeStrategy(strategy);
          if (result?.success) break;
        }
      }

      return this.formatResponse(result);

    } catch (error) {
      return {
        status: 500,
        body: {
          success: false,
          error: error.message,
          recoveryAttempted: this.attemptRecovery(error)
        }
      };
    }
  }
}

// Updated Maintenance Schedule Controller
export const getMaintenanceSchedule = async (req, res) => {
  try {
    const { shipId, type } = req.query;
    
    if (!shipId || !type) {
      return res.status(400).json({ message: 'Missing required parameters' });
    }

    const agent = new MaintenanceAgent(shipId, type);
    const { status, body } = await agent.execute();
    
    // Format LLM response
    if (body.predictionSource === 'llm_fallback') {
      body.llmAnalysis = body.analysis;
      delete body.analysis;
      
      // For unsupported types, include a note
      if (!['Engine Overhaul'].includes(type)) {
        body.note = 'Using AI analysis for unsupported maintenance type';
      }
    }
    
    return res.status(status).json(body);

  } catch (error) {
    logger.error('Maintenance schedule error:', error);
    res.status(500).json({ 
      status: 'error',
      message: 'Internal server error',
      details: error.message 
    });
  }
};