import mongoose from 'mongoose';

const routeSchema = new mongoose.Schema({
  shipId: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'Ship', 
    required: true 
    }, // Reference to the ship
  departure: {
    port: { 
        type: String, 
        required: true 
    }, // Departure port name
    timestamp: { 
        type: Date, 
        required: true 
    }, // Departure timestamp
    coordinates: {
      lat: { 
        type: Number, 
        required: true 
    }, // Latitude of departure port
      lng: { 
        type: Number, 
        required: true 
    } // Longitude of departure port
    }
  },
  destination: {
    port: { 
        type: String, 
        required: true 
    }, // Destination port name
    estimatedTimestamp: { 
        type: Date, 
        required: true 
    }, // Estimated arrival time
    coordinates: {
      lat: { 
        type: Number, 
        required: true 
    }, // Latitude of destination port
      lng: { 
        type: Number, 
        required: true 
    } // Longitude of destination port
    }
  },
  waypoints: [{
    coordinates: {
      lat: { 
        type: Number, 
        required: true 
    }, // Latitude of waypoint
      lng: { 
        type: Number, 
        required: true 
    } // Longitude of waypoint
    },
    timestamp: { 
        type: Date, 
        required: true 
    } // Timestamp at the waypoint
  }],
  distance: { 
    type: Number, 
    required: true 
    }, // Total route distance in nautical miles
  status: { 
    type: String, 
    required: true 
    }, // Route status (e.g., planned, in-progress)
  cargoWeight: { 
    type: Number, 
    required: true
     }, // Cargo weight in tons
  estimatedDuration: { 
    type: Number, 
    required: true 
    }, // Estimated duration in hours
  actualDuration: { 
    type: Number 
    }, // Actual duration in hours (after completion)
  weatherConditions: {
    averageWindSpeed: { 
        type: Number 
    }, // Average wind speed during the route
    averageWaveHeight: { 
        type: Number 
    }, // Average wave height during the route
    predominantDirection: { 
        type: String 
    } // Predominant weather direction
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

export default mongoose.model('Route', routeSchema);
