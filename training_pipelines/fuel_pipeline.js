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

// TrainingProgressCallback class remains unchanged
class TrainingProgressCallback {
    constructor(jobData, maxEpochs) {
        this.jobData = jobData;
        this.maxEpochs = maxEpochs;
        this.bestValLoss = Infinity;
        this.bestWeights = null;
        this.model = null;
    }
    
    setParams(params) {}
    setModel(model) { this.model = model; }
    onTrainBegin(logs) {}
    onEpochBegin(epoch, logs) {}
    onBatchBegin(batch, logs) {}
    onBatchEnd(batch, logs) {}
    
    onTrainEnd(logs) {
        try {
            if (this.bestWeights && this.model) {
                this.model.setWeights(this.bestWeights);
                this.bestWeights.forEach(w => w.dispose());
            }
            
            if (this.jobData?.progressCallback && typeof this.jobData.progressCallback === 'function') {
                this.jobData.progressCallback(100, 'Training completed');
            }
        } catch (err) {
            logger.warn('Train end callback error:', err);
        }
    }
    
    async onEpochEnd(epoch, logs) {
        try {
            if (logs && logs.val_loss < this.bestValLoss) {
                this.bestValLoss = logs.val_loss;
                this.bestWeights = this.model.getWeights().map(w => w.clone());
            }
            
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
}

// Custom Learning Rate Scheduler
class CustomLearningRateScheduler extends tf.Callback {
    constructor(schedule, initialLr, dataSize) {
        super();
        this.schedule = schedule;
        this.initialLr = initialLr;
        this.dataSize = dataSize;
        this.currentEpoch = 0;
    }
    
    onEpochBegin(epoch) {
        this.currentEpoch = epoch;
        const decayRate = this.dataSize < 30 ? 0.85 : 0.95;
        const newLr = this.initialLr * Math.pow(decayRate, epoch);
        
        const optimizer = this.model.optimizer;
        if (optimizer && 'learningRate' in optimizer) {
            optimizer.learningRate = newLr;
        }
        return Promise.resolve();
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
                if (weatherFetchFailures > MAX_FAILED_WEATHER_REQUESTS) {
                    throw new Error('Too many weather request failures, aborting');
                }
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
        let processedData = applyPreprocessing(filteredData, preprocessingParams);
        
        // --- Small Data Handling ---
        const originalDataSize = processedData.length;
        if (originalDataSize < 30) {
            logger.info(`Small dataset detected (${originalDataSize} samples). Applying augmentation...`);
            processedData = augmentDataWithNoise(processedData, 0.15);
            logger.info(`Data augmented to ${processedData.length} samples`);
        }
        
        tf.util.shuffle(processedData);
        
        // --- Data Splitting ---
        const effectiveTestRatio = originalDataSize < 30 ? 0.15 : TEST_SET_RATIO;
        const splitIndex = Math.floor(processedData.length * (1 - effectiveTestRatio));
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
        
        // --- Adaptive Model Architecture ---
        logger.info('Building model with adaptive complexity...');
        
        const getLayerSizes = (dataSize) => {
            if (dataSize < 20) return [16, 8];
            if (dataSize < 50) return [32, 16];
            return [128, 64, 32];
        };
        
        const layerSizes = getLayerSizes(originalDataSize);
        logger.info(`Using layer architecture: ${layerSizes.join(', ')} for ${originalDataSize} original samples`);
        
        const l2Strength = originalDataSize < 30 ? 0.05 : 
                          originalDataSize < 50 ? 0.02 : 0.01;
        
        model = tf.sequential();
        
        // Input layer
        model.add(tf.layers.dense({
            inputShape: [X_train.shape[1]],
            units: layerSizes[0],
            activation: 'relu',
            kernelInitializer: 'heNormal',
            kernelRegularizer: tf.regularizers.l2({ l2: l2Strength })
        }));
        model.add(tf.layers.batchNormalization());
        
        const dropoutRate = originalDataSize < 30 ? 0.2 : 0.3;
        model.add(tf.layers.dropout({ rate: dropoutRate }));
        
        // Hidden layers
        for (let i = 1; i < layerSizes.length; i++) {
            model.add(tf.layers.dense({
                units: layerSizes[i],
                activation: 'relu',
                kernelRegularizer: tf.regularizers.l2({ l2: l2Strength })
            }));
            
            if (i < layerSizes.length - 1) {
                model.add(tf.layers.batchNormalization());
                model.add(tf.layers.dropout({ rate: dropoutRate }));
            }
        }
        
        // Output layer
        model.add(tf.layers.dense({ units: 1 }));

        // --- Model Compilation ---
        const initialLearningRate = originalDataSize < 30 ? 0.002 : INITIAL_LEARNING_RATE;
        
        model.compile({
            optimizer: tf.train.adam(initialLearningRate),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        model.summary();

        // --- Custom Learning Rate Scheduler ---
        const lrScheduler = new CustomLearningRateScheduler(
            (epoch, lr) => {
                const decayRate = originalDataSize < 30 ? 0.85 : 0.95;
                return initialLearningRate * Math.pow(decayRate, epoch);
            },
            initialLearningRate,
            originalDataSize
        );
        
        // --- Adaptive Early Stopping ---
        const earlyStoppingPatience = originalDataSize < 30 ? 
                                     Math.min(10, EARLY_STOPPING_PATIENCE) : 
                                     EARLY_STOPPING_PATIENCE;
        
        const minDelta = originalDataSize < 30 ? 100 : 1000;
        
        // --- Model Training ---
        logger.info(`Training model with adaptive parameters (patience: ${earlyStoppingPatience}, minDelta: ${minDelta})...`);
        const progressCallback = new TrainingProgressCallback(jobData, MAX_EPOCHS);
        
        const effectiveValidationSplit = originalDataSize < 30 ? 0.1 : VALIDATION_SPLIT;
        
        const adaptiveBatchSize = Math.min(
            BATCH_SIZE, 
            Math.max(4, Math.floor(X_train.shape[0] / 4))
        );
        
        const history = await model.fit(X_train, y_train, {
            epochs: MAX_EPOCHS,
            batchSize: adaptiveBatchSize,
            validationSplit: effectiveValidationSplit,
            callbacks: [
                tf.callbacks.earlyStopping({
                    monitor: 'val_loss',
                    patience: earlyStoppingPatience,
                    minDelta: minDelta,
                    // restoreBestWeights: true
                }),
                lrScheduler,
                progressCallback
            ],
            verbose: 0
        });
        
        // --- Bayesian Evaluation for Small Datasets ---
        let uncertaintyEstimate = null;
        if (originalDataSize < 30) {
            logger.info('Performing Bayesian uncertainty estimation for small dataset...');
            const bayesianResults = await performBayesianInference(model, X_test);
            uncertaintyEstimate = {
                meanUncertainty: mean(bayesianResults.uncertainties),
                maxUncertainty: Math.max(...bayesianResults.uncertainties)
            };
            logger.info(`Uncertainty estimate: mean=${uncertaintyEstimate.meanUncertainty.toFixed(2)}, max=${uncertaintyEstimate.maxUncertainty.toFixed(2)}`);
        }

        // --- Model Evaluation ---
        const evalResult = model.evaluate(X_test, y_test);
        const testLoss = evalResult[0].dataSync()[0];
        const testRMSE = Math.sqrt(testLoss);
        
        logger.info(`Test RMSE: ${testRMSE.toFixed(2)}`);

        // --- Model Saving ---
        logger.info('Saving model...');
        await model.save(FUEL_MODEL_SAVE_PATH);
        
        const metadata = {
            originalDataSize,
            augmentedDataSize: processedData.length,
            architecture: layerSizes,
            l2Strength,
            dropoutRate,
            uncertaintyEstimate
        };
        
        await fs.writeFile(
            path.join(MODEL_OUTPUT_DIR, 'model_metadata.json'), 
            JSON.stringify(metadata, null, 2)
        );
        
        trainingSuccess = true;
        return {
            status: 'success',
            modelPath: FUEL_MODEL_SAVE_PATH,
            testRMSE,
            trainingDataSize: X_train.shape[0],
            originalDataSize,
            weatherFetchFailures,
            finalEpoch: history.epoch.length,
            finalLoss: history.history.loss[history.history.loss.length - 1],
            finalValLoss: history.history.val_loss[history.history.val_loss.length - 1],
            uncertaintyEstimate
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

// --- Helper Functions for Small Data Handling ---
const augmentDataWithNoise = (data, noiseFactor = 0.15, multiplier = 3) => {
    const augmented = [...data];
    const targetMean = mean(data.map(d => d.target));
    const targetStd = standardDeviation(data.map(d => d.target));
    
    data.forEach(sample => {
        for (let i = 0; i < multiplier; i++) {
            const newFeatures = [...sample.features];
            
            for (let j = 0; j < newFeatures.length; j++) {
                newFeatures[j] += newFeatures[j] * (Math.random() - 0.5) * noiseFactor;
            }
            
            const newTarget = sample.target * (1 + (Math.random() - 0.5) * (noiseFactor / 2));
            const boundedTarget = Math.max(
                targetMean - 2 * targetStd,
                Math.min(newTarget, targetMean + 2 * targetStd)
            );
            
            augmented.push({
                features: newFeatures,
                target: boundedTarget
            });
        }
    });
    
    return augmented;
};

const performBayesianInference = async (model, input, numSamples = 10) => {
    const predictions = [];
    
    for (let i = 0; i < numSamples; i++) {
        const predTensor = model.predict(input, {training: true});
        predictions.push(Array.from(await predTensor.data()));
        predTensor.dispose();
    }
    
    const results = {
        means: [],
        uncertainties: []
    };
    
    for (let i = 0; i < predictions[0].length; i++) {
        const samples = predictions.map(p => p[i]);
        results.means.push(mean(samples));
        results.uncertainties.push(standardDeviation(samples));
    }
    
    return results;
};