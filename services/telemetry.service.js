import  Telemetry  from '../models/Telemetry.js';
import logger from '../utils/logger.js';

class TelemetryService {
  static async getLatest(shipId) {
    try {
      const telemetry = await Telemetry.findOne({ shipId })
        .sort({ timestamp: -1 })
        .lean();
      
      if (!telemetry) {
        logger.warn(`No telemetry data found for ship ${shipId}`);
        return null;
      }

      // Normalize telemetry data
      return {
        timestamp: telemetry.timestamp,
        engineHours: telemetry.engineHours,
        temperature: telemetry.temperature,
        vibration: telemetry.vibration,
        pressure: telemetry.pressure,
        oilQuality: telemetry.oilQuality,
        fuelConsumption: telemetry.fuelConsumption
      };
    } catch (error) {
      logger.error(`Failed to fetch telemetry for ship ${shipId}:`, error);
      throw new Error('Telemetry service unavailable');
    }
  }

  static async getHistorical(shipId, hours = 24) {
    try {
      const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
      return await Telemetry.find({
        shipId,
        timestamp: { $gte: cutoff }
      }).sort({ timestamp: 1 });
    } catch (error) {
      logger.error(`Failed to fetch historical telemetry:`, error);
      throw new Error('Historical telemetry unavailable');
    }
  }

  static async getHealthStatus(shipId) {
    const telemetry = await this.getLatest(shipId);
    if (!telemetry) return 'unknown';
    
    const thresholds = {
      temperature: { warn: 85, critical: 95 },
      vibration: { warn: 6.5, critical: 8.0 },
      pressure: { warn: 105, critical: 115 }
    };

    const status = {
      temperature: telemetry.temperature > thresholds.temperature.critical ? 'critical' :
                 telemetry.temperature > thresholds.temperature.warn ? 'warning' : 'normal',
      vibration: telemetry.vibration > thresholds.vibration.critical ? 'critical' :
                telemetry.vibration > thresholds.vibration.warn ? 'warning' : 'normal',
      pressure: telemetry.pressure > thresholds.pressure.critical ? 'critical' :
               telemetry.pressure > thresholds.pressure.warn ? 'warning' : 'normal'
    };

    return {
      overall: Object.values(status).includes('critical') ? 'critical' :
              Object.values(status).includes('warning') ? 'warning' : 'normal',
      details: status,
      lastUpdated: telemetry.timestamp
    };
  }
}

export default TelemetryService;