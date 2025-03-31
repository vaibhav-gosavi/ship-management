// services/prediction.service.js
import fs from 'fs/promises';
import path from 'path';
import { Types } from 'mongoose';
import * as tf from '@tensorflow/tfjs-node';
import logger from '../utils/logger.js';
import Ship from '../models/ship.model.js';
import Maintenance from '../models/maintenance.model.js';
import FuelLog from '../models/fuel.model.js';
import weatherService from './weatherService.js'; // Import weather service

// --- Configuration ---
const MODEL_OUTPUT_DIR = path.resolve(process.cwd(), 'tfjs_models_store');
// Fuel Model Paths
const FUEL_MODEL_LOAD_PATH = `file://${path.join(MODEL_OUTPUT_DIR, 'fuel-model', 'model.json')}`;
const FUEL_PREPROCESSING_PARAMS_PATH = path.join(MODEL_OUTPUT_DIR, 'fuel_preprocessing_params.json');
// Maintenance Model Paths
// To (simpler version):
const MAINTENANCE_MODEL_PATH = `file://${path.join(MODEL_OUTPUT_DIR, 'maintenance-model-Engine_Overhaul-v3')}`;
const MAINTENANCE_PARAMS_PATH = path.join(MODEL_OUTPUT_DIR, 'maintenance_preprocessing_params-Engine_Overhaul-v3.json');

// --- Caching ---
let loadedFuelModel = null;
let loadedFuelPreprocessingParams = null;
const loadedMaintenanceModels = {}; // { 'TypeKey': { model, preprocessing } }

// --- Helper: Load Model & Params ---
const loadModelAndParamsGeneric = async (modelCache, paramsCache, modelLoadPath, paramsLoadPath, modelTypeLogPrefix = 'Model') => {
    let modelToLoad = modelCache;
    let paramsToLoad = paramsCache;

    // Check if already loaded
    if (modelToLoad && paramsToLoad) {
        return { model: modelToLoad, preprocessing: paramsToLoad };
    }

    // Reset cache in case only one part was loaded before error
    modelToLoad = null;
    paramsToLoad = null;

    try {
        logger.info(`[${modelTypeLogPrefix}] Loading preprocessing parameters from ${paramsLoadPath}...`);
        const paramsContent = await fs.readFile(paramsLoadPath, 'utf-8');
        paramsToLoad = JSON.parse(paramsContent);
        logger.info(`[${modelTypeLogPrefix}] Preprocessing parameters loaded.`);

        logger.info(`[${modelTypeLogPrefix}] Loading TF.js model from ${modelLoadPath}...`);
        modelToLoad = await tf.loadLayersModel(modelLoadPath);
        logger.info(`[${modelTypeLogPrefix}] TF.js model loaded.`);

        // Warm up the model (optional, helps with first prediction latency)
        // tf.tidy(() => {
        //     const featureCount = paramsToLoad.numericalFeatures.length + calculateCategoricalWidth(paramsToLoad.encoders); // Helper needed
        //     const dummyInput = tf.zeros([1, featureCount]);
        //     modelToLoad.predict(dummyInput);
        //     logger.info(`[${modelTypeLogPrefix}] Model warmed up.`);
        // });

        return { model: modelToLoad, preprocessing: paramsToLoad };

    } catch (error) {
        logger.error(`[${modelTypeLogPrefix}] Failed to load model or parameters:`, error);
        // Do not cache partially loaded state
        throw new Error(`Could not load ${modelTypeLogPrefix} model/params`);
    }
};

// --- Load Fuel Model and Params (uses generic loader) ---
const loadFuelModelAndParams = async () => {
    const result = await loadModelAndParamsGeneric(
        loadedFuelModel,
        loadedFuelPreprocessingParams,
        FUEL_MODEL_LOAD_PATH,
        FUEL_PREPROCESSING_PARAMS_PATH,
        'Fuel'
    );
    // Store in cache
    loadedFuelModel = result.model;
    loadedFuelPreprocessingParams = result.preprocessing;
    return result;
};

// --- Load Maintenance Model and Params (uses generic loader) ---
const loadMaintenanceModelAndParams = async (maintenanceType) => {
    const typeKey = maintenanceType.replace(/\s+/g, '_');
    const modelCache = loadedMaintenanceModels[typeKey]?.model;
    const paramsCache = loadedMaintenanceModels[typeKey]?.preprocessing;

    // Use the direct paths without any modification
    const paramsPath = MAINTENANCE_PARAMS_PATH;
    const modelLoadPath = `${MAINTENANCE_MODEL_PATH}/model.json`;

    // Add debug logging
    logger.debug(`Loading model from: ${modelLoadPath}`);
    logger.debug(`Loading params from: ${paramsPath}`);

    const result = await loadModelAndParamsGeneric(
        modelCache,
        paramsCache,
        modelLoadPath,
        paramsPath,
        `Maintenance [${maintenanceType}]`
    );

    // Store in cache
    loadedMaintenanceModels[typeKey] = {
        model: result.model,
        preprocessing: result.preprocessing
    };
    return result;
};

// --- Helper: Apply Preprocessing for Prediction Input ---
const applyPreprocessingForInput = (inputData, params) => {
    const { numericalFeatures, categoricalFeatures, imputation, scalers, encoders } = params;
    const featureVector = [];

    // Check if all expected features are present in inputData
    numericalFeatures.forEach(col => {
        if (!(col in inputData)) logger.warn(`Missing numerical feature in inputData: ${col}. Imputation will be used.`);
    });
    categoricalFeatures.forEach(col => {
         if (!(col in inputData)) logger.warn(`Missing categorical feature in inputData: ${col}. Default encoding (all zeros) will be used.`);
    });


    for (const col of numericalFeatures) {
        const scaler = scalers[col];
        if (!scaler) throw new Error(`Prediction Preprocessing Error: Scaler missing for ${col}`);
        let value = inputData[col]; // Get value, could be null/undefined
        if (value == null || isNaN(value)) {
            value = imputation[col]; // Apply imputation value calculated during training
        }
        // Check scaler params before dividing
        if (scaler.stdDev === 0 || isNaN(scaler.stdDev)) {
             logger.warn(`Scaler stdDev is invalid for ${col}. Using 0 for scaled value.`);
             featureVector.push(0);
        } else {
             featureVector.push((value - scaler.mean) / scaler.stdDev);
        }
    }

    for (const col of categoricalFeatures) {
        const encoder = encoders[col];
        if (!encoder) throw new Error(`Prediction Preprocessing Error: Encoder missing for ${col}`);
        const encoded = new Array(encoder.categories.length).fill(0);
        const value = inputData[col]; // Get value, could be null/undefined
        const index = encoder.mapping[value]; // Find index in mapping
        if (index !== undefined) { // Only set to 1 if category was seen during training
            encoded[index] = 1;
        }
        featureVector.push(...encoded);
    }
    return featureVector;
};


// --- Predict Fuel Consumption ---
export const predictFuelConsumption = async (shipId, routeDetails) => {
    // Validate shipId
    if (!shipId || !Types.ObjectId.isValid(shipId)) {
        logger.error(`Invalid ship ID format: ${shipId}`);
        return null;
    }

    // Convert to ObjectId
    const shipObjectId = new Types.ObjectId(shipId);
    let inputTensor;

    try {
        // Debug: Log the ID conversion
        logger.debug(`Original ID: ${shipId}, Converted ID: ${shipObjectId}`);

        // 1. Load model and preprocessing params
        const { model, preprocessing } = await loadFuelModelAndParams();

        // 2. Fetch Ship with proper error handling
        const ship = await Ship.findById(shipObjectId).lean();
        
        if (!ship) {
            // Debug: Check what ships exist
            const sampleShips = await Ship.find({}).limit(3).lean();
            logger.warn('Sample ships in DB:', sampleShips.map(s => s._id.toString()));
            throw new Error(`Ship ${shipId} not found. Converted to: ${shipObjectId}`);
        }

        logger.debug(`Found ship: ${ship._id} (${ship.name || 'unnamed'})`);

        // 3. Get weather forecast
        let forecastWeather = null;
        if (routeDetails.departureLat && routeDetails.departureLon) {
            try {
                forecastWeather = await weatherService.getForecastWeather(
                    routeDetails.departureLat, 
                    routeDetails.departureLon
                );
                logger.debug('Weather forecast:', forecastWeather);
            } catch (weatherError) {
                logger.warn('Weather forecast failed:', weatherError.message);
            }
        }

        // 4. Prepare input data
        const inputData = {
            ship_type: ship.type,
            ship_buildYear: ship.buildYear,
            ship_capacity_weight: ship.capacity?.weight || 0,
            ship_engine_power: ship.engine?.power || 0,
            ship_length: ship.dimensions?.length || 0,
            route_distance: routeDetails.distance,
            route_cargoWeight: routeDetails.cargoWeight || 0,
            route_actual_duration_hours: routeDetails.estimatedDuration,
            weather_wind_speed: forecastWeather?.windSpeed || 0,
            weather_temp: forecastWeather?.temperature || 20 // Default to 20°C if missing
        };

        logger.debug('Input data for prediction:', inputData);

        // 5. Preprocess and predict
        let predictedFuel = 0;
        await tf.tidy(() => {
            const preparedFeatures = preprocessInput(inputData, preprocessing);
            inputTensor = tf.tensor2d([preparedFeatures]);
            const prediction = model.predict(inputTensor);
            predictedFuel = prediction.dataSync()[0];
        });

        logger.info(`Predicted fuel for ${shipId}: ${predictedFuel.toFixed(2)} units`);
        return Math.max(0, predictedFuel);

    } catch (error) {
        logger.error('Prediction failed:', {
            error: error.message,
            shipId,
            convertedId: shipObjectId.toString(),
            stack: error.stack
        });
        return null;
    }
};
function preprocessInput(inputData, preprocessingParams) {
    const features = [];
    
    // Process numerical features
    for (const col of preprocessingParams.numericalFeatures) {
        const scaler = preprocessingParams.scalers[col];
        let value = inputData[col] ?? preprocessingParams.imputation[col];
        features.push((value - scaler.mean) / scaler.stdDev);
    }
    
    // Process categorical features
    for (const col of preprocessingParams.categoricalFeatures) {
        const encoder = preprocessingParams.encoders[col];
        const encoded = new Array(encoder.categories.length).fill(0);
        const value = inputData[col];
        
        if (value !== undefined && encoder.mapping[value] !== undefined) {
            encoded[encoder.mapping[value]] = 1;
        }
        features.push(...encoded);
    }
    
    return features;
}

// --- Predict Maintenance RUL ---
export const predictNextMaintenanceRUL = async (shipId, maintenanceType) => {
    let inputTensor;
    try {
        const { model, preprocessing } = await loadMaintenanceModelAndParams(maintenanceType);

        // 1. Fetch latest maintenance of this type
        const lastMaintenance = await Maintenance.findOne({
            shipId: shipId, type: maintenanceType, engineHoursAtMaintenance: { $exists: true, $ne: null }
        }).sort({ engineHoursAtMaintenance: -1 }).lean();
        if (!lastMaintenance) {
             return { predictedRULHours: null, error: 'No prior maintenance data' };
        }
        const engineHoursAtLastMaintenance = lastMaintenance.engineHoursAtMaintenance;

        // 2. Fetch Current Engine Hours (CRITICAL)
        const latestFuelLog = await FuelLog.findOne({ shipId: shipId }).sort({ timestamp: -1 }).lean();
        if (!latestFuelLog || latestFuelLog.engineHours === undefined) {
             return { predictedRULHours: null, error: 'Cannot determine current engine hours' };
        }
        const currentTotalEngineHours = latestFuelLog.engineHours;

        if (currentTotalEngineHours <= engineHoursAtLastMaintenance) {
             return { predictedRULHours: 0, hoursSinceLast: 0, warning: 'Current hours not sufficiently past last maintenance' };
        }
        const hoursSinceLast = currentTotalEngineHours - engineHoursAtLastMaintenance;

        // 3. Fetch Ship details
        const ship = await Ship.findById(shipId).lean();
        if (!ship) throw new Error(`Ship with ID ${shipId} not found.`);

        // 4. Construct input feature object (matching training)
        const inputData = {
            ship_type: ship.type,
            ship_buildYear: ship.buildYear,
            ship_capacity_weight: ship.capacity?.weight,
            ship_engine_power: ship.engine?.power,
            // Add other features model was trained on
        };

        // 5. Transform input
        const preparedFeatures = applyPreprocessingForInput(inputData, preprocessing);

        // 6. Convert to Tensor and Predict
        let predictedIntervalHours = 0;
        await tf.tidy(() => {
            inputTensor = tf.tensor2d([preparedFeatures]);
            const predictionTensor = model.predict(inputTensor);
            predictedIntervalHours = predictionTensor.dataSync()[0];
        });

        // 7. Calculate RUL
        const predictedRULHours = predictedIntervalHours - hoursSinceLast;

        logger.info(`[${maintenanceType}] Ship ${shipId}: Last @ ${engineHoursAtLastMaintenance} hrs. Current: ${currentTotalEngineHours}. Hrs Since Last: ${hoursSinceLast.toFixed(0)}. Predicted Interval: ${predictedIntervalHours.toFixed(0)} hrs. Predicted RUL: ${predictedRULHours.toFixed(0)} hrs.`);

        return {
             predictedRULHours: Math.max(0, predictedRULHours),
             predictedIntervalHours: predictedIntervalHours,
             hoursSinceLast: hoursSinceLast,
             engineHoursAtLastMaintenance: engineHoursAtLastMaintenance,
             currentEngineHoursEstimate: currentTotalEngineHours
         };

    } catch (error) {
        logger.error(`Error during maintenance prediction for Ship ${shipId}, Type ${maintenanceType}:`, error);
        return { predictedRULHours: null, error: error.message };
    }
    // No finally needed if using tidy
};

// --- Clear Cache ---
export const clearModelCache = () => {
    // Fuel
    loadedFuelModel = null; // Let GC handle disposal of model object
    loadedFuelPreprocessingParams = null;
    logger.info('TF.js Fuel model cache cleared.');
    // Maintenance
    for (const key in loadedMaintenanceModels) {
         delete loadedMaintenanceModels[key];
    }
    logger.info('TF.js Maintenance model cache cleared.');
};

export default {
    predictFuelConsumption,
    predictNextMaintenanceRUL,
    loadFuelModelAndParams, // Keep exports if needed for testing/debugging
    loadMaintenanceModelAndParams,
    clearModelCache
};