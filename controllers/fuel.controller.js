import FuelLog from '../models/fuel.model.js';
import Ship from '../models/ship.model.js';
import Route from '../models/route.model.js';
import predictionService from '../services/prediction.service.js';
export const addFuelLog = async (req, res) => {
  try {
    const {
      shipId,
      routeId,
      timestamp,
      fuelType,
      quantity,
      consumptionRate,
      engineHours,
      rpm,
      speed,
      weatherConditions,
      notes
    } = req.body;

    // Validate required fields
    if (!shipId || !routeId || !fuelType || !quantity || !consumptionRate || 
        !engineHours || !rpm || !speed) {
      return res.status(400).json({ 
        message: 'Please provide all required fuel log details' 
      });
    }

    console.log('shipId:', shipId);
    console.log('routeId:', routeId);
    
    
    // Verify ship and route exist
    const shipExists = await Ship.exists({ _id: shipId });
    const routeExists = await Route.exists({ _id: routeId });

    console.log('shipExists:', shipExists);
    console.log('routeExists:', routeExists);
    

    if (!shipExists) {
      return res.status(404).json({ message: `Ship with ID ${shipId} not found` });
    }
    if (!routeExists) {
      return res.status(404).json({ message: `Route with ID ${routeId} not found` });
    }

    // Create new fuel log
    const fuelLog = new FuelLog({
      shipId,
      routeId,
      timestamp: timestamp || new Date(),
      fuelType,
      quantity,
      consumptionRate,
      engineHours,
      rpm,
      speed,
      weatherConditions,
      notes
    });

    const savedFuelLog = await fuelLog.save();
    res.status(201).json(savedFuelLog);

  } catch (error) {
    console.error('Error adding fuel log:', error);
    res.status(500).json({ 
      message: 'Error recording fuel log',
      error: error.message 
    });
  }
};

// {
//   "shipId": "507f1f77bcf86cd799439011",
//   "routeId": "5f8d0d55b54764421b7156da",
//   "timestamp": "2023-05-15T08:30:00Z",
//   "fuelType": "Diesel",
//   "quantity": 1500,
//   "consumptionRate": 125,
//   "engineHours": 2450,
//   "rpm": 1800,
//   "speed": 12.5,
//   "weatherConditions": {
//     "windSpeed": 15,
//     "waveHeight": 2.5,
//     "temperature": 22
//   },
//   "notes": "Calm seas, normal consumption"
// }



export const getFuelEstimate = async (req, res, next) => {
  const {
      shipid, shipId,  // Handle both cases
      distance, 
      cargoWeight, 
      estimatedDuration,
      departureLat, 
      departureLon
  } = req.query;

  // --- Validation ---
  const errors = [];

  // Check presence and type of each parameter
  if (!shipid && !shipId) {
      errors.push('shipId is required');
  }
  const finalShipId = shipid || shipId;

  if (!distance || isNaN(parseFloat(distance))) {
      errors.push('distance must be a valid number');
  }

  if (!cargoWeight || isNaN(parseFloat(cargoWeight))) {
      errors.push('cargoWeight must be a valid number');
  }

  if (!estimatedDuration || isNaN(parseFloat(estimatedDuration))) {
      errors.push('estimatedDuration must be a valid number');
  }

  if (!departureLat || isNaN(parseFloat(departureLat))) {
      errors.push('departureLat must be a valid number');
  }

  if (!departureLon || isNaN(parseFloat(departureLon))) {
      errors.push('departureLon must be a valid number');
  }

  if (errors.length > 0) {
      return res.status(400).json({ 
          message: 'Validation failed',
          errors: errors
      });
  }

  // --- Processing ---
  try {
      const routeDetails = {
          distance: parseFloat(distance),
          cargoWeight: parseFloat(cargoWeight),
          estimatedDuration: parseFloat(estimatedDuration),
          departureLat: parseFloat(departureLat),
          departureLon: parseFloat(departureLon),
      };

      const predictedFuel_1 = await predictionService.predictFuelConsumption(finalShipId, routeDetails);
      
      const predictedFuel = predictedFuel_1 * 1000;

      if (predictedFuel === null) {
          return res.status(404).json({ message: 'Ship not found or prediction failed' });
      }
      
      res.status(200).json({ 
          shipId: finalShipId,
          estimatedFuelConsumption: predictedFuel,
          units: 'liters'
      });

  } catch (error) {
      console.error('Error fetching fuel estimate:', error);
      res.status(500).json({ message: 'Error fetching fuel estimate', error: error.message });
  }
};