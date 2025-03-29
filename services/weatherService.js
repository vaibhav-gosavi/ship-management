// services/weatherService.js
import axios from 'axios';
import NodeCache from 'node-cache';
import logger from '../utils/logger.js';
import 'dotenv/config';

// --- Configuration ---
const WEATHER_API_KEY = process.env.OPENWEATHERMAP_API_KEY;
const WEATHER_API_BASE_URL = 'https://api.openweathermap.org/data/2.5';
const HISTORICAL_DATA_ENABLED = process.env.HISTORICAL_DATA_ENABLED === 'true';
const cache = new NodeCache({ stdTTL: 3600, checkperiod: 600 });

// Validate API key on startup
if (!WEATHER_API_KEY) {
    logger.error('FATAL: OPENWEATHERMAP_API_KEY environment variable not set!');
    throw new Error('Weather API configuration error');
}

// Helper function for common weather data extraction
const extractWeatherData = (data, source) => {
    if (!data || !data.wind || !data.main) {
        throw new Error('Invalid weather data structure received');
    }

    return {
        timestamp: new Date(data.dt * 1000),
        temperature: data.main.temp,
        windSpeed: data.wind.speed * 1.94384, // Convert m/s to knots
        windDirection: data.wind.deg,
        humidity: data.main.humidity,
        pressure: data.main.pressure,
        waveHeight: null, // Will remain null unless using marine API
        source: source,
    };
};

// Fallback weather data generator
const getDummyWeatherData = () => {
    return {
        timestamp: new Date(),
        temperature: 15 + Math.random() * 10,
        windSpeed: 5 + Math.random() * 15,
        windDirection: Math.floor(Math.random() * 360),
        humidity: 60 + Math.random() * 30,
        pressure: 1010 + Math.random() * 10,
        waveHeight: null,
        source: 'dummy_data',
    };
};

/**
 * Enhanced weather forecast with better error handling and logging
 */
const getForecastWeather = async (lat, lon) => {
    const cacheKey = `forecast-${lat.toFixed(2)}-${lon.toFixed(2)}`;
    const cachedData = cache.get(cacheKey);
    if (cachedData) {
        logger.debug(`Cache hit for forecast: ${cacheKey}`);
        return cachedData;
    }

    logger.debug(`Fetching forecast for coordinates: ${lat}, ${lon}`);
    
    try {
        const response = await axios.get(`${WEATHER_API_BASE_URL}/weather`, {
            params: {
                lat: lat,
                lon: lon,
                appid: WEATHER_API_KEY,
                units: 'metric'
            },
            timeout: 5000 // 5 second timeout
        });

        const weatherInfo = extractWeatherData(response.data, 'openweathermap_current');
        cache.set(cacheKey, weatherInfo);
        return weatherInfo;
    } catch (error) {
        logger.error(`Forecast API error:`, {
            coordinates: { lat, lon },
            error: error.message,
            status: error.response?.status,
            data: error.response?.data
        });

        // Return cached data even if stale if available
        const staleData = cache.get(cacheKey, true);
        if (staleData) {
            logger.warn('Using stale cached forecast data due to API failure');
            return staleData;
        }

        logger.warn('No cached data available - generating dummy weather data');
        return getDummyWeatherData();
    }
};

/**
 * Historical weather with proper plan verification and fallbacks
 */
const getHistoricalWeather = async (lat, lon, timestamp) => {
    if (!HISTORICAL_DATA_ENABLED) {
        logger.debug('Historical data disabled - falling back to current weather');
        return getForecastWeather(lat, lon);
    }

    const unixTimestamp = Math.floor(new Date(timestamp).getTime() / 1000);
    const cacheKey = `hist-${lat.toFixed(2)}-${lon.toFixed(2)}-${unixTimestamp}`;
    
    const cachedData = cache.get(cacheKey);
    if (cachedData) return cachedData;

    logger.debug(`Fetching historical weather for ${lat}, ${lon} at ${timestamp}`);
    
    try {
        const response = await axios.get(`${WEATHER_API_BASE_URL}/onecall/timemachine`, {
            params: {
                lat,
                lon,
                dt: unixTimestamp,
                appid: WEATHER_API_KEY,
                units: 'metric'
            },
            timeout: 5000
        });

        if (!response.data?.current) {
            throw new Error('No current data in historical response');
        }

        const weatherInfo = extractWeatherData(response.data.current, 'openweathermap_historical');
        cache.set(cacheKey, weatherInfo);
        return weatherInfo;
    } catch (error) {
        logger.error(`Historical weather API error:`, {
            coordinates: { lat, lon },
            timestamp,
            error: error.message,
            status: error.response?.status,
            data: error.response?.data
        });

        // Fallback strategies in order of preference:
        // 1. Try current weather if historical isn't available
        logger.warn('Falling back to current weather data');
        try {
            const current = await getForecastWeather(lat, lon);
            return {
                ...current,
                source: 'fallback_current_to_historical',
                originalTimestamp: timestamp,
            };
        } catch (fallbackError) {
            logger.error('Current weather fallback also failed:', fallbackError.message);
            return getDummyWeatherData();
        }
    }
};

export default {
    getForecastWeather,
    getHistoricalWeather,
};