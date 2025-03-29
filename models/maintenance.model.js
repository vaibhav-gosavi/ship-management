import mongoose from 'mongoose';

const maintenanceSchema = new mongoose.Schema({
  shipId: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'Ship', 
    required: true 
    }, // Reference to the ship
  type: { 
    type: String, 
    required: true 
    }, // Type of maintenance (e.g., engine, hull)
  description: { 
    type: String, 
    required: true 
    }, // Description of the maintenance
  performedAt: { 
    type: Date, 
    required: true 
    }, // Date when maintenance was performed
  nextDue: { 
    type: Date 
    }, // Predicted next maintenance date
  engineHoursAtMaintenance: { 
    type: Number 
    }, // Engine hours at the time of maintenance
  partsReplaced: [
    {
    name: { 
        type: String, 
        required: true 
    }, // Name of the replaced part
    serialNumber: { 
        type: String 
    }, // Serial number of the replaced part
    lifeExpectancy: { 
        type: Number 
    } // Life expectancy of the part in hours
  }],
  cost: { 
    type: Number, 
    required: true 
    }, // Cost of the maintenance
  technician: { 
    type: String, 
    required: true 
    }, // Technician who performed the maintenance
  createdAt: { 
    type: Date, 
    default: Date.now 
    }, // Record creation timestamp
  updatedAt: { 
    type: Date, 
    default: Date.now 
} // Record update timestamp
});

export default mongoose.model('Maintenance', maintenanceSchema);
