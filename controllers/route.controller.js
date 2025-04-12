// Route controller logic
import Route from '../models/route.model.js';
import Ship from '../models/ship.model.js';

export const addRoute = async (req, res) => {
  try {
    // Data for the new route comes from the request body
    const {
      shipId,
      departure, // { port, timestamp, coordinates: { lat, lng } }
      destination, // { port, estimatedTimestamp, coordinates: { lat, lng } }
      waypoints, // Optional array of { coordinates: { lat, lng }, timestamp }
      distance,
      status,
      cargoWeight,
      estimatedDuration,
      weatherConditions // Optional { averageWindSpeed, averageWaveHeight, predominantDirection }
      // actualDuration is usually set upon completion
    } = req.body;

    // --- Basic Validation ---
    if (
      !shipId || !departure || !destination || !distance || !status ||
      cargoWeight === undefined || estimatedDuration === undefined ||
      !departure.port || !departure.timestamp || !departure.coordinates || !departure.coordinates.lat || !departure.coordinates.lng ||
      !destination.port || !destination.estimatedTimestamp || !destination.coordinates || !destination.coordinates.lat || !destination.coordinates.lng
    ) {
      return res.status(400).json({ message: 'Please provide all required route details (shipId, departure, destination, distance, status, cargoWeight, estimatedDuration)' });
    }
     // Validate waypoints if provided
     if (waypoints && Array.isArray(waypoints)) {
        for (const wp of waypoints) {
            if (!wp.coordinates || !wp.coordinates.lat || !wp.coordinates.lng || !wp.timestamp) {
                return res.status(400).json({ message: 'Invalid waypoint structure. Each waypoint needs coordinates (lat, lng) and timestamp.' });
            }
        }
     }
    // --- End Basic Validation ---


    // Verify that the referenced Ship actually exists
    const shipExists = await Ship.findById(shipId);
    if (!shipExists) {
      return res.status(404).json({ message: `Ship with ID ${shipId} not found.` });
    }

    // Create a new route instance
    const newRoute = new Route({
      shipId,
      departure,
      destination,
      waypoints, // Will be included if present in req.body, otherwise undefined (Mongoose handles this)
      distance,
      status,
      cargoWeight,
      estimatedDuration,
      weatherConditions, // Will be included if present
      // createdAt and updatedAt handled by Mongoose
      // actualDuration is typically null or undefined initially
    });

    // Save the new route to the database
    const savedRoute = await newRoute.save();

    // Respond with the created route data
    res.status(201).json(savedRoute);

  } catch (error) {
    console.error('Error adding route:', error);
    // Handle potential validation errors from Mongoose
    if (error.name === 'ValidationError') {
        return res.status(400).json({ message: 'Validation Error', errors: error.errors });
    }
    res.status(500).json({ message: 'Server error while adding route' });
  }
};

// this is the example request body
// {
//   "shipId": "60d5ecf1a5a8f813b4a1b4f0", // <- IMPORTANT: Replace with an ACTUAL ObjectId of an existing Ship in your database
//   "departure": {
//     "port": "Port of Singapore",
//     "timestamp": "2024-08-01T10:00:00.000Z", // ISO 8601 format UTC is recommended
//     "coordinates": {
//       "lat": 1.2646,
//       "lng": 103.8198
//     }
//   },
//   "destination": {
//     "port": "Port of Los Angeles",
//     "estimatedTimestamp": "2024-08-20T18:00:00.000Z", // ISO 8601 format
//     "coordinates": {
//       "lat": 33.7292,
//       "lng": -118.2620
//     }
//   },
//   "waypoints": [ // Optional: Include if the route has specific intermediate points
//     {
//       "coordinates": {
//         "lat": 21.1458, // Example waypoint: Near Hawaii
//         "lng": -157.8228
//       },
//       "timestamp": "2024-08-10T12:00:00.000Z" // Estimated time at waypoint
//     }
//   ],
//   "distance": 7500, // Total route distance in nautical miles
//   "status": "Planned", // Initial status (e.g., "Planned", "Scheduled")
//   "cargoWeight": 150000, // Cargo weight in tons for this specific route
//   "estimatedDuration": 464, // Estimated duration in hours (e.g., distance / avg speed)
//   "weatherConditions": { // Optional: Include if you have forecast averages
//     "averageWindSpeed": 12, // Knots
//     "averageWaveHeight": 1.8, // Meters
//     "predominantDirection": "W"
//   }
//   // "actualDuration" is usually omitted on creation, set later upon completion.
// }


export const addShip = async (req, res) => {
  try {
    // Data for the new ship comes from the request body
    const {
      name,
      imoNumber,
      type,
      buildYear,
      capacity, // { weight, volume }
      engine,   // { type, power, fuelType }
      dimensions // { length, width, draft }
    } = req.body;

    // Basic validation example (Check if required fields are present)
    // Consider using a validation library like Joi or express-validator for robust validation
    if (!name || !imoNumber || !type || !buildYear || !capacity || !engine || !dimensions) {
      return res.status(400).json({ message: 'Please provide all required ship details' });
    }
    if(!capacity.weight || !capacity.volume || !engine.type || !engine.power || !engine.fuelType || !dimensions.length || !dimensions.width || !dimensions.draft) {
       return res.status(400).json({ message: 'Please provide all nested required ship details (capacity, engine, dimensions)' });
    }

    // Check if a ship with the same IMO number already exists
    const existingShip = await Ship.findOne({ imoNumber });
    if (existingShip) {
      return res.status(400).json({ message: `Ship with IMO number ${imoNumber} already exists.` });
    }

    // Create a new ship instance
    const newShip = new Ship({
      name,
      imoNumber,
      type,
      buildYear,
      capacity,
      engine,
      dimensions,
      // createdAt and updatedAt will be handled by Mongoose defaults
    });

    // Save the new ship to the database
    const savedShip = await newShip.save();

    // Respond with the created ship data
    res.status(201).json(savedShip);

  } catch (error) {
    console.error('Error adding ship:', error);
     // Handle potential duplicate key error specifically (though checked above, race conditions possible)
     if (error.code === 11000) {
         return res.status(400).json({ message: `Duplicate field value entered: ${Object.keys(error.keyValue)}` });
     }
    res.status(500).json({ message: 'Server error while adding ship' });
  }
};

// this is the example response 
// {
//   "name": "Ever Ace",
//   "imoNumber": "9893890", // Must be unique
//   "type": "Container Ship",
//   "buildYear": 2021,
//   "capacity": {
//     "weight": 235579, // DWT (Deadweight Tonnage) in tons
//     "volume": 23992  // Assuming TEU capacity here, adjust if it's cubic meters as per schema
//                      // If volume is cubic meters, e.g., 200000
//   },
//   "engine": {
//     "type": "Wärtsilä X92", // Example engine type
//     "power": 70950,      // Example power in kW
//     "fuelType": "Very Low Sulphur Fuel Oil (VLSFO)" // Example fuel type
//   },
//   "dimensions": {
//     "length": 399.9,   // meters
//     "width": 61.5,     // meters
//     "draft": 17.0      // meters (maximum draft)
//   }
// }



// Add this to your existing route.controller.js

export const completeRoute = async (req, res) => {
  try {
    const { id } = req.params;
    const { actualDuration, weatherConditions } = req.body;

    if (!actualDuration) {
      return res.status(400).json({ 
        message: 'Actual duration is required to complete route' 
      });
    }

    // Find and update the route
    const updatedRoute = await Route.findByIdAndUpdate(
      id,
      {
        status: 'completed',
        actualDuration,
        weatherConditions: weatherConditions || undefined,
        updatedAt: new Date()
      },
      { new: true, runValidators: true }
    );

    if (!updatedRoute) {
      return res.status(404).json({ message: `Route with ID ${id} not found` });
    }

    res.json(updatedRoute);

  } catch (error) {
    console.error('Error completing route:', error);
    res.status(500).json({ 
      message: 'Error completing route',
      error: error.message 
    });
  }
};

// {
//   "actualDuration": 48.5,
//   "weatherConditions": {
//     "averageWindSpeed": 18,
//     "averageWaveHeight": 3.2,
//     "predominantDirection": "NE"
//   }
// }

export const listShips = async (req, res) => {
  try {
    const { 
      page = 1, 
      limit = 10, 
      sort = 'name', 
      order = 'asc',
      type,
      buildYearMin,
      buildYearMax
    } = req.query;

    // Build query
    const query = {};
    if (type) query.type = type;
    if (buildYearMin || buildYearMax) {
      query.buildYear = {};
      if (buildYearMin) query.buildYear.$gte = parseInt(buildYearMin);
      if (buildYearMax) query.buildYear.$lte = parseInt(buildYearMax);
    }

    // Calculate skip for pagination
    const skip = (parseInt(page) - 1) * parseInt(limit);

    // Build sort object
    const sortObj = {};
    sortObj[sort] = order === 'desc' ? -1 : 1;

    // Execute query with pagination
    const ships = await Ship.find(query)
      .sort(sortObj)
      .skip(skip)
      .limit(parseInt(limit))
      .lean();

    // Get total count for pagination
    const totalShips = await Ship.countDocuments(query);

    // Send response
    res.status(200).json({
      ships,
      pagination: {
        currentPage: parseInt(page),
        totalPages: Math.ceil(totalShips / parseInt(limit)),
        totalShips,
        hasNextPage: skip + ships.length < totalShips,
        hasPrevPage: parseInt(page) > 1
      }
    });

  } catch (error) {
    console.error('Error listing ships:', error);
    res.status(500).json({ message: 'Server error while listing ships' });
  }
};