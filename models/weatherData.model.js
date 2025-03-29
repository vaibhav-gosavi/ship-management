import mongoose from 'mongoose';

const weatherDataSchema = new mongoose.Schema({
  coordinates: {
    lat: { 
        type: Number, 
        required: true 
    }, // Latitude of the location
    lng: { 
        type: Number, 
        required: true 
    } // Longitude of the location
  },
  timestamp: { 
    type: Date, 
    required: true 
    }, // Timestamp of the weather data
  windSpeed: { 
    type: Number, 
    required: true 
    }, // Wind speed in knots
  windDirection: { 
    type: Number, 
    required: true 
    }, // Wind direction in degrees
  waveHeight: { 
    type: Number, 
    required: true 
    }, // Wave height in meters
  temperature: { 
    type: Number, 
    required: true 
    }, // Temperature in Celsius
  precipitation: { 
    type: Number 
    }, // Precipitation in mm/h
  visibility: { 
    type: Number 
    }, // Visibility in nautical miles
  forecastSource: { 
    type: String, 
    required: true 
    }, // Source of the weather forecast
  expiresAt: { 
    type: Date, 
    required: true 
    } // Expiry time of the cached data
});

export default mongoose.model('WeatherData', weatherDataSchema);
