import mongoose from 'mongoose';

const shipSchema = new mongoose.Schema({
  name: { 
    type: String,
    required: true 
    }, // Ship name for identification
  imoNumber: { 
    type: String,
    unique: true, 
    required: true 
    }, // Unique identifier for the ship
  type: { 
    type: String, 
    required: true 
    }, // Type of ship (e.g., Container, Tanker)
  buildYear: { 
    type: Number, 
    required: true 
    }, // Year the ship was built
  capacity: {
    weight: { 
        type: Number, 
        required: true 
    }, // Maximum weight capacity in tons
    volume: { 
        type: Number, 
        required: true 
    } // Maximum volume capacity in cubic meters
  },
  engine: {
    type: { 
        type: String, 
        required: true 
    }, // Engine type
    power: { 
        type: Number, 
        required: true 
    }, // Engine power in kW
    fuelType: { 
        type: String, 
        required: true 
    } // Fuel type used by the engine
  },
  dimensions: {
    length: { 
        type: Number, 
        required: true 
    }, // Length of the ship in meters
    width: { 
        type: Number, 
        required: true 
    }, // Width of the ship in meters
    draft: { 
        type: Number, 
        required: true 
    } // Draft of the ship in meters
  },
  createdAt: { 
    type: Date, 
    default: Date.now 
    }, // Record creation timestamp
  updatedAt: { 
    type: Date, 
    default: Date.now 
} // Record update timestamp
});

export default mongoose.model('Ship', shipSchema);
