import mongoose from "mongoose";

const fuelLogSchema = new mongoose.Schema({
  shipId: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: "Ship", 
    required: true 
    }, // Reference to the ship
  routeId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Route",
    required: true,
  }, // Reference to the route
  timestamp: {
    type: Date, 
    required: true 
    }, // Timestamp of the fuel log
  fuelType: { 
    type: String, 
    required: true 
    }, // Type of fuel used
  quantity: { 
    type: Number, 
    required: true 
    }, // Quantity of fuel in liters
  consumptionRate: { 
    type: Number, 
    required: true 
    }, // Fuel consumption rate in liters/hour
  engineHours: { 
    type: Number, 
    required: true 
    }, // Engine operating hours
  rpm: { 
    type: Number, 
    required: true 
    }, // Engine revolutions per minute
  speed: { 
    type: Number, 
    required: true 
    }, // Ship speed in knots
  weatherConditions: {
    windSpeed: { 
        type: Number 
    }, // Wind speed during the log
    waveHeight: { 
        type: Number 
    }, // Wave height during the log
    temperature: { 
        type: Number 
    }, // Temperature during the log
  },
  notes: { 
    type: String
 }, // Additional notes
  createdAt: { 
    type: Date, 
    default: Date.now 
  }, // Record update timestamp 
});

export default mongoose.model("FuelLog", fuelLogSchema);
