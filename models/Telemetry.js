import mongoose from 'mongoose';

const telemetrySchema = new mongoose.Schema({
  shipId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Ship',
    required: true
  },
  timestamp: {
    type: Date,
    default: Date.now,
    required: true
  },
  engineHours: {
    type: Number,
    required: true
  },
  temperature: Number,    // in °C
  vibration: Number,      // in mm/s
  pressure: Number,       // in psi
  oilQuality: Number,     // 0-100%
  fuelConsumption: Number // in liters/hour
}, { timestamps: true });

telemetrySchema.index({ shipId: 1, timestamp: -1 });

const Telemetry = mongoose.model('Telemetry', telemetrySchema);

export default Telemetry;