import mongoose from 'mongoose';
import fs from 'fs/promises';
import path from 'path';
import * as tf from '@tensorflow/tfjs-node';
import { mean, median, standardDeviation } from 'simple-statistics';
import logger from '../utils/logger.js';
import Ship from '../models/ship.model.js';
import Route from '../models/route.model.js';
import FuelLog from '../models/fuel.model.js';
import weatherService from '../services/weatherService.js';

// --- Configuration ---
const MODEL_OUTPUT_DIR = path.resolve(process.cwd(), 'tfjs_models_store');
const FUEL_MODEL_SAVE_PATH = `file://${path.join(MODEL_OUTPUT_DIR, 'fuel-model')}`;
const PREPROCESSING_PARAMS_PATH = path.join(MODEL_OUTPUT_DIR, 'fuel_preprocessing_params.json');
const TEST_SET_RATIO = 0.2;
const VALIDATION_SPLIT = 0.15;
const MIN_TRAINING_SAMPLES = 10;
const MAX_FAILED_WEATHER_REQUESTS = 5;
const INITIAL_LEARNING_RATE = 0.001;
const MAX_EPOCHS = 200;
const BATCH_SIZE = 8;
const EARLY_STOPPING_PATIENCE = 20;

// --- Helper Functions ---
const getImputationParams = (data, column, strategy = 'median') => {
    const values = data.map(row => row[column]).filter(v => v != null && !isNaN(v));
    if (values.length === 0) {
        logger.warn(`No valid values found for imputation in column: ${column}. Using 0.`);
        return 0;
    }
    return strategy === 'mean' ? mean(values) : median(values);
};

const getOneHotEncodingParams = (data, column) => {
    const categories = [...new Set(data.map(row => row[column]).filter(v => v != null))].sort();
    const mapping = {};
    categories.forEach((value, index) => { mapping[value] = index; });
    return { categories, mapping };
};

const getScalingParams = (data, column) => {
    const values = data.map(row => row[column]).filter(v => v != null && !isNaN(v));
    const meanVal = mean(values);
    const stdDevVal = standardDeviation(values);
    return { 
        mean: meanVal, 
        stdDev: (stdDevVal === 0 || isNaN(stdDevVal)) ? 1 : stdDevVal 
    };
};

const applyPreprocessing = (data, params) => {
    return data.map(row => {
        const features = [];
        
        // Process numerical features
        for (const col of params.numericalFeatures) {
            const scaler = params.scalers[col];
            const value = row[col] ?? params.imputation[col];
            features.push((value - scaler.mean) / scaler.stdDev);
        }
        
        // Process categorical features
        for (const col of params.categoricalFeatures) {
            const encoder = params.encoders[col];
            const encoded = new Array(encoder.categories.length).fill(0);
            if (row[col] != null && encoder.mapping[row[col]] !== undefined) {
                encoded[encoder.mapping[row[col]]] = 1;
            }
            features.push(...encoded);
        }
        
        return { 
            features, 
            target: row[params.targetVariable] 
        };
    }).filter(item => !item.features.some(isNaN));
};

const validateRouteData = (route) => {
    const requiredFields = {
        coordinates: route.departure?.coordinates?.lat && route.departure?.coordinates?.lng,
        timestamp: route.departure?.timestamp,
        distance: route.distance > 0,
        duration: route.actualDuration > 0
    };
    
    return Object.values(requiredFields).every(Boolean);
};

// Updated TrainingProgressCallback class with all required methods
class TrainingProgressCallback {
    constructor(jobData, maxEpochs) {
        this.jobData = jobData;
        this.maxEpochs = maxEpochs;
        this.bestValLoss = Infinity;
        this.bestWeights = null;
        this.model = null;
    }
    
    // Required callback methods
    setParams(params) {
        // Implementation required but can be empty
    }
    
    setModel(model) {
        this.model = model;
    }
    
    onTrainBegin(logs) {
        // Optional: Called when training starts
    }
    
    onTrainEnd(logs) {
        // Restore best weights when training ends
        try {
            if (this.bestWeights && this.model) {
                this.model.setWeights(this.bestWeights);
                this.bestWeights.forEach(w => w.dispose());
            }
            
            // Complete progress reporting
            if (this.jobData?.progressCallback && typeof this.jobData.progressCallback === 'function') {
                this.jobData.progressCallback(100, 'Training completed');
            }
        } catch (err) {
            logger.warn('Train end callback error:', err);
        }
    }
    
    onEpochBegin(epoch, logs) {
        // Optional: Called at the beginning of each epoch
    }
    
    async onEpochEnd(epoch, logs) {
        try {
            // Track best weights
            if (logs && logs.val_loss < this.bestValLoss) {
                this.bestValLoss = logs.val_loss;
                this.bestWeights = this.model.getWeights().map(w => w.clone());
            }
            
            // Report progress
            if (this.jobData?.progressCallback && typeof this.jobData.progressCallback === 'function') {
                const progress = Math.min(90, Math.floor((epoch / this.maxEpochs) * 90));
                await this.jobData.progressCallback(
                    progress, 
                    `Epoch ${epoch + 1}/${this.maxEpochs} - Loss: ${logs.loss.toFixed(2)}, Val Loss: ${logs.val_loss.toFixed(2)}`
                );
            }
        } catch (err) {
            logger.warn('Progress callback error:', err);
        }
    }
    
    onBatchBegin(batch, logs) {
        // Optional: Called at the beginning of each batch
    }
    
    onBatchEnd(batch, logs) {
        // Optional: Called at the end of each batch
    }
}


// --- Main Pipeline Function ---
export const runFuelModelTrainingPipeline = async (jobData) => {
    logger.info('Starting Fuel Model Training Pipeline...');
    const tensorsToDispose = [];
    let trainingSuccess = false;
    let model;

    try {
        // --- Data Collection ---
        logger.info('Fetching training data...');
        const [routes, ships, fuelAggregate] = await Promise.all([
            Route.find({
                status: 'Completed',
                'departure.coordinates.lat': { $exists: true },
                'departure.coordinates.lng': { $exists: true },
                'departure.timestamp': { $exists: true },
                distance: { $gt: 0 },
                actualDuration: { $gt: 0 },
                cargoWeight: { $exists: true }
            }).lean(),
            Ship.find().lean(),
            FuelLog.aggregate([
                { $match: { quantity: { $gt: 0 } } },
                { $group: { _id: '$routeId', totalFuelConsumed: { $sum: '$quantity' } } }
            ])
        ]);

        if (!routes.length) {
            logger.warn('No valid routes found for training');
            return { status: 'skipped', reason: 'No route data' };
        }

        // --- Data Preparation ---
        const validRoutes = routes.filter(validateRouteData);
        const shipsMap = new Map(ships.map(s => [s._id.toString(), s]));
        const fuelMap = new Map(fuelAggregate.map(f => [f._id.toString(), f.totalFuelConsumed]));

        // --- Feature Engineering ---
        let weatherFetchFailures = 0;
        const combinedData = await Promise.all(validRoutes.map(async route => {
            const ship = shipsMap.get(route.shipId.toString());
            const totalFuel = fuelMap.get(route._id.toString());
            
            if (!ship || !totalFuel) return null;

            let weather = null;
            try {
                weather = await weatherService.getHistoricalWeather(
                    route.departure.coordinates.lat,
                    route.departure.coordinates.lng,
                    route.departure.timestamp
                );
            } catch (error) {
                weatherFetchFailures++;
                logger.warn(`Weather fetch error: ${error.message}`);
            }

            return {
                ship_type: ship.type,
                ship_buildYear: ship.buildYear,
                ship_capacity_weight: ship.capacity?.weight,
                ship_engine_power: ship.engine?.power,
                ship_length: ship.dimensions?.length,
                route_distance: route.distance,
                route_cargoWeight: route.cargoWeight,
                route_actual_duration_hours: route.actualDuration,
                weather_wind_speed: weather?.windSpeed || 0,
                weather_temp: weather?.temperature || 20,
                target_fuel_consumed: totalFuel
            };
        }));

        const filteredData = combinedData.filter(Boolean);
        if (filteredData.length < MIN_TRAINING_SAMPLES) {
            const msg = `Insufficient data (${filteredData.length} < ${MIN_TRAINING_SAMPLES})`;
            logger.warn(msg);
            return { status: 'skipped', reason: msg };
        }

        // --- Preprocessing ---
        logger.info('Preprocessing data...');
        const categoricalFeatures = ['ship_type'];
        const numericalFeatures = [
            'ship_buildYear', 'ship_capacity_weight', 'ship_engine_power',
            'ship_length', 'route_distance', 'route_cargoWeight',
            'route_actual_duration_hours', 'weather_wind_speed', 'weather_temp'
        ];

        const preprocessingParams = {
            numericalFeatures,
            categoricalFeatures,
            targetVariable: 'target_fuel_consumed',
            imputation: {},
            scalers: {},
            encoders: {}
        };

        // Calculate preprocessing parameters
        numericalFeatures.forEach(col => {
            preprocessingParams.imputation[col] = getImputationParams(filteredData, col);
            preprocessingParams.scalers[col] = getScalingParams(filteredData, col);
        });

        categoricalFeatures.forEach(col => {
            preprocessingParams.encoders[col] = getOneHotEncodingParams(filteredData, col);
        });

        // Save preprocessing parameters
        await fs.mkdir(MODEL_OUTPUT_DIR, { recursive: true });
        await fs.writeFile(PREPROCESSING_PARAMS_PATH, JSON.stringify(preprocessingParams, null, 2));

        // Apply preprocessing
        const processedData = applyPreprocessing(filteredData, preprocessingParams);
        tf.util.shuffle(processedData);
        
        // --- Data Splitting ---
        const splitIndex = Math.floor(processedData.length * (1 - TEST_SET_RATIO));
        const trainData = processedData.slice(0, splitIndex);
        const testData = processedData.slice(splitIndex);

        // Create tensors
        const createTensor = (data, name) => {
            const tensor = tf.tensor2d(data);
            tensorsToDispose.push(tensor);
            return tensor;
        };

        const X_train = createTensor(trainData.map(d => d.features), 'X_train');
        const y_train = createTensor(trainData.map(d => [d.target]), 'y_train');
        const X_test = createTensor(testData.map(d => d.features), 'X_test');
        const y_test = createTensor(testData.map(d => [d.target]), 'y_test');

        // --- Model Architecture ---
        logger.info('Building model...');
        model = tf.sequential();
        
        // Input layer with regularization
        model.add(tf.layers.dense({
            inputShape: [X_train.shape[1]],
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
        }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        // Hidden layers
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
        }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }));
        
        // Output layer
        model.add(tf.layers.dense({ units: 1 }));

        // --- Model Compilation ---
        model.compile({
            optimizer: tf.train.adam(INITIAL_LEARNING_RATE),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        model.summary();

        // --- Model Training ---
        logger.info('Training model...');
        const progressCallback = new TrainingProgressCallback(jobData, MAX_EPOCHS);

const history = await model.fit(X_train, y_train, {
    epochs: MAX_EPOCHS,
    batchSize: BATCH_SIZE,
    validationSplit: VALIDATION_SPLIT,
    callbacks: [
        tf.callbacks.earlyStopping({
            monitor: 'val_loss',
            patience: EARLY_STOPPING_PATIENCE,
            minDelta: 1000
        }),
        progressCallback
    ],
    verbose: 0
});
        // --- Model Evaluation ---
        const evalResult = model.evaluate(X_test, y_test);
        const testLoss = evalResult[0].dataSync()[0];
        const testRMSE = Math.sqrt(testLoss);
        
        logger.info(`Test RMSE: ${testRMSE.toFixed(2)}`);

        // --- Model Saving ---
        logger.info('Saving model...');
        await model.save(FUEL_MODEL_SAVE_PATH);
        
        trainingSuccess = true;
        return {
            status: 'success',
            modelPath: FUEL_MODEL_SAVE_PATH,
            testRMSE,
            trainingDataSize: X_train.shape[0],
            weatherFetchFailures,
            finalEpoch: history.epoch.length,
            finalLoss: history.history.loss[history.history.loss.length - 1],
            finalValLoss: history.history.val_loss[history.history.val_loss.length - 1]
        };

    } catch (error) {
        logger.error('Pipeline failed:', error);
        throw error;
    } finally {
        // Cleanup tensors
        tensorsToDispose.forEach(t => {
            try { t?.dispose(); } catch (e) { logger.warn('Tensor disposal error:', e); }
        });
        
        // Cleanup model if it exists
        if (model) {
            try { model.dispose(); } catch (e) { logger.warn('Model disposal error:', e); }
        }
        
        logger.info(`Pipeline ${trainingSuccess ? 'completed successfully' : 'failed'}`);
    }
};