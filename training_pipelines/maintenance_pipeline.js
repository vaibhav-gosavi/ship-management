// training_pipelines/maintenance_pipeline.js
import mongoose from 'mongoose';
import fs from 'fs/promises';
import path from 'path';
import * as tf from '@tensorflow/tfjs-node';
import * as ss from 'simple-statistics'; // Using simple-statistics
import logger from '../utils/logger.js';
import Ship from '../models/ship.model.js';
import Maintenance from '../models/maintenance.model.js';

// --- Configuration ---
const MODEL_OUTPUT_DIR = path.resolve(process.cwd(), 'tfjs_models_store');
const MAINTENANCE_MODEL_SAVE_PATH_PREFIX = `file://${path.join(MODEL_OUTPUT_DIR, 'maintenance-model')}`;
const PREPROCESSING_PARAMS_PATH_PREFIX = path.join(MODEL_OUTPUT_DIR, 'maintenance_preprocessing_params');
const TEST_SET_RATIO = 0.2;
const MIN_TRAINING_SAMPLES = 15;
// --- !! UPDATED: Only training 'Engine Overhaul' based on latest logs !! ---
const TARGET_MAINTENANCE_TYPES = ['Engine Overhaul'];
// -----------------------------------------------------------------------------

// --- Helper Functions (Using simple-statistics) ---
const getImputationParams = (data, column, strategy = 'median') => {
    const values = data.map(row => row[column]).filter(v => v != null && !isNaN(v));
    if (values.length === 0) {
        logger.warn(`No valid values for imputation in column ${column}. Using 0.`);
        return 0;
    }
    // Use simple-statistics functions
    return strategy === 'mean' ? ss.mean(values) : ss.median(values);
};

const getOneHotEncodingParams = (data, column) => {
    const validData = data.filter(row => row[column] != null);
    if (validData.length === 0) {
        logger.warn(`No valid categories for encoding in column ${column}. Using empty mapping.`);
        return { categories: [], mapping: {} };
    }
    const categories = [...new Set(validData.map(row => row[column]))].sort();
    const mapping = {};
    categories.forEach((value, index) => { mapping[value] = index; });
    return { categories, mapping };
};

const getScalingParams = (data, column) => {
    const values = data.map(row => row[column]).filter(v => v != null && !isNaN(v));
    if (values.length < 2) {
        logger.warn(`Insufficient values (<2) for scaling column ${column}. Using mean=0, std=1.`);
        return { mean: 0, stdDev: 1 };
    }
    const meanVal = ss.mean(values);
    // Use sampleStandardDeviation for robustness with smaller datasets
    const stdDevVal = ss.sampleStandardDeviation(values);
    return {
        mean: meanVal,
        stdDev: (stdDevVal === 0 || isNaN(stdDevVal)) ? 1 : stdDevVal
    };
};

const applyPreprocessing = (data, params) => {
    const { numericalFeatures, categoricalFeatures, imputation, scalers, encoders } = params;
    const processedData = [];

    for (const row of data) {
        try {
            const featureVector = [];

            // Process numerical features
            for (const col of numericalFeatures) {
                const scaler = scalers[col];
                if (!scaler) throw new Error(`Preprocessing Error: Scaler missing for ${col}`);
                // Use nullish coalescing for potentially missing values before imputation lookup
                let value = row[col] ?? imputation[col];
                // Handle cases where imputation value itself might be needed if row[col] is null/undefined
                if (value == null || isNaN(value)) {
                    value = imputation[col];
                }
                // Check scaler params validity
                if (scaler.stdDev === 0 || isNaN(scaler.stdDev)) {
                    logger.warn(`Scaler stdDev is invalid for ${col} during apply. Using 0 for scaled value.`);
                    featureVector.push(0);
                } else {
                    featureVector.push((value - scaler.mean) / scaler.stdDev);
                }
            }

            // Process categorical features
            for (const col of categoricalFeatures) {
                const encoder = encoders[col];
                if (!encoder) throw new Error(`Preprocessing Error: Encoder missing for ${col}`);
                const encoded = new Array(encoder.categories.length).fill(0);
                const value = row[col]; // Can be null/undefined
                if (value != null && encoder.mapping[value] !== undefined) {
                    encoded[encoder.mapping[value]] = 1;
                }
                featureVector.push(...encoded);
            }

            processedData.push({
                features: featureVector,
                target: row[params.targetVariable]
            });
        } catch (error) {
            logger.warn(`Error processing row during applyPreprocessing: ${error.message}`, { rowData: row }); // Log row data for debugging
        }
    }

    return processedData;
};

// --- Data Validation ---
const validateMaintenanceRecord = (record, shipInfo) => {
    if (!shipInfo) {
        logger.debug(`Validation failed - missing ship info for ship ${record.shipId}`);
        return { valid: false, reason: 'missing_ship_info' };
    }
    
    if (!record.engineHoursAtMaintenance || isNaN(record.engineHoursAtMaintenance)) {
        logger.debug(`Validation failed - invalid engine hours for record ${record._id}, value: ${record.engineHoursAtMaintenance}`);
        return { valid: false, reason: 'invalid_engine_hours' };
    }
    
    // Make checks safer with optional chaining
    if (!shipInfo.type) {
        logger.debug(`Validation failed - missing ship type for ship ${shipInfo._id}`);
        return { valid: false, reason: 'missing_ship_type' };
    }
    
    if (!shipInfo.buildYear) {
        logger.debug(`Validation failed - missing build year for ship ${shipInfo._id}`);
        return { valid: false, reason: 'missing_build_year' };
    }
    
    if (!shipInfo.capacity?.weight) {
        logger.debug(`Validation failed - missing capacity weight for ship ${shipInfo._id}`);
        return { valid: false, reason: 'missing_capacity_weight' };
    }
    
    if (!shipInfo.engine?.power) {
        logger.debug(`Validation failed - missing engine power for ship ${shipInfo._id}`);
        return { valid: false, reason: 'missing_engine_power' };
    }
    
    return { valid: true, reason: null };
};

// --- Main Pipeline Function ---
export const runMaintenanceModelTrainingPipeline = async (jobData) => {
    logger.info('Starting Maintenance Model Training Pipeline...');
    const results = {};
    const tensorsToDispose = [];
    let overallSuccess = false; // Track if at least one model succeeds

    try {
        // Ensure DB connection (optional check, might rely on worker's connection)
        if (mongoose.connection.readyState !== 1) {
            logger.warn(`MongoDB connection not ready (state: ${mongoose.connection.readyState}). Attempting connection (may fail if worker disconnected).`);
            // Avoid connecting here if worker should manage connection
            // await mongoose.connect(process.env.MONGODB_URI);
        }

        for (const maintenanceType of TARGET_MAINTENANCE_TYPES) {
            logger.info(`--- Training model for Maintenance Type: [${maintenanceType}] ---`);
            // Updated versioning for model and params filenames
            const modelSavePath = `${MAINTENANCE_MODEL_SAVE_PATH_PREFIX}-${maintenanceType.replace(/\s+/g, '_')}-v2`;
            const paramsSavePath = `${PREPROCESSING_PARAMS_PATH_PREFIX}-${maintenanceType.replace(/\s+/g, '_')}-v2.json`;

            try {
                // --- 1. Data Collection ---
                logger.info(`[${maintenanceType}] Fetching maintenance records...`);
                const maintenanceRecords = await Maintenance.find({
                    type: maintenanceType,
                    engineHoursAtMaintenance: { $exists: true, $ne: null },
                    shipId: { $exists: true }
                })
                // Sort also by engineHours to handle potential timestamp ties correctly
                .sort({ shipId: 1, engineHoursAtMaintenance: 1, performedAt: 1 })
                .lean();

                logger.info(`[${maintenanceType}] Found ${maintenanceRecords.length} maintenance records`);

                if (maintenanceRecords.length < 2) {
                    logger.warn(`[${maintenanceType}] Insufficient maintenance records (< 2). Skipping.`);
                    results[maintenanceType] = {
                        status: 'skipped',
                        reason: 'insufficient_records',
                        count: maintenanceRecords.length
                    };
                    continue;
                }

                const shipIds = [...new Set(maintenanceRecords.map(m => m.shipId.toString()))];
                logger.info(`[${maintenanceType}] Found ${shipIds.length} unique ships with maintenance records`);
                
                const ships = await Ship.find({ _id: { $in: shipIds } }).lean();
                logger.info(`[${maintenanceType}] Retrieved ${ships.length} ship records`);
                
                const shipsMap = new Map(ships.map(s => [s._id.toString(), s]));

                // --- 2. Data Preparation ---
                logger.info(`[${maintenanceType}] Preparing training data...`);
                let combinedData = [];
                
                // Track validation failures
                const validationFailures = {
                    missing_ship_info: 0,
                    invalid_engine_hours: 0,
                    missing_ship_type: 0,
                    missing_build_year: 0,
                    missing_capacity_weight: 0,
                    missing_engine_power: 0,
                    invalid_interval: 0
                };
                
                // Group records by ship
                const recordsByShip = {};
                for (const record of maintenanceRecords) {
                    const shipIdStr = record.shipId.toString();
                    if (!recordsByShip[shipIdStr]) {
                        recordsByShip[shipIdStr] = [];
                    }
                    recordsByShip[shipIdStr].push(record);
                }
                
                // Count ships with multiple records
                const shipsWithMultipleRecords = Object.keys(recordsByShip).filter(
                    shipId => recordsByShip[shipId].length >= 2
                ).length;
                
                logger.info(`[${maintenanceType}] Ships with multiple maintenance records: ${shipsWithMultipleRecords}`);

                for (const shipIdStr in recordsByShip) {
                    const shipRecords = recordsByShip[shipIdStr];
                    const shipInfo = shipsMap.get(shipIdStr);
                    
                    if (shipRecords.length < 2) {
                        logger.debug(`Ship ${shipIdStr} has only one maintenance record. Need at least 2 to calculate interval.`);
                        continue;
                    }
                    
                    logger.debug(`Processing ${shipRecords.length} maintenance records for ship ${shipIdStr}`);

                    for (let i = 1; i < shipRecords.length; i++) {
                        const prevMaint = shipRecords[i - 1];
                        const currentMaint = shipRecords[i];

                        // Validate both records
                        const prevValidation = validateMaintenanceRecord(prevMaint, shipInfo);
                        if (!prevValidation.valid) {
                            validationFailures[prevValidation.reason]++;
                            continue;
                        }
                        
                        const currentValidation = validateMaintenanceRecord(currentMaint, shipInfo);
                        if (!currentValidation.valid) {
                            validationFailures[currentValidation.reason]++;
                            continue;
                        }

                        const interval = currentMaint.engineHoursAtMaintenance - prevMaint.engineHoursAtMaintenance;
                        
                        if (interval <= 0) {
                            // Log more context for invalid intervals
                            logger.debug(`Invalid interval (${interval}) for ship ${shipIdStr}. PrevHrs: ${prevMaint.engineHoursAtMaintenance} (@${prevMaint.performedAt}), CurrHrs: ${currentMaint.engineHoursAtMaintenance} (@${currentMaint.performedAt})`);
                            validationFailures.invalid_interval++;
                            continue;
                        }

                        // Valid interval found, add to training data
                        combinedData.push({
                            ship_type: shipInfo.type,
                            ship_buildYear: shipInfo.buildYear,
                            ship_capacity_weight: shipInfo.capacity.weight,
                            ship_engine_power: shipInfo.engine.power,
                            target_engineHoursInterval: interval
                        });
                        
                        // Log successful interval for debugging
                        logger.debug(`Valid interval: ${interval} hours for ship ${shipIdStr} (${prevMaint.engineHoursAtMaintenance} -> ${currentMaint.engineHoursAtMaintenance})`);
                    }
                } // End interval calculation loop

                // Log validation failures summary
                logger.info(`[${maintenanceType}] Validation failure summary: ${JSON.stringify(validationFailures)}`);
                logger.info(`[${maintenanceType}] Valid intervals generated: ${combinedData.length}`);
                
                if (combinedData.length < MIN_TRAINING_SAMPLES) {
                    logger.warn(`[${maintenanceType}] Insufficient valid intervals (${combinedData.length} < ${MIN_TRAINING_SAMPLES}). Skipping training.`);
                    results[maintenanceType] = {
                        status: 'skipped',
                        reason: 'insufficient_valid_intervals',
                        count: combinedData.length,
                        required: MIN_TRAINING_SAMPLES,
                        validationFailures
                    };
                    continue;
                }

                // --- 3. Preprocessing ---
                logger.info(`[${maintenanceType}] Configuring preprocessing...`);
                const categoricalFeatures = ['ship_type'];
                const numericalFeatures = ['ship_buildYear', 'ship_capacity_weight', 'ship_engine_power'];
                const targetVariable = 'target_engineHoursInterval';
                
                // Calculate preprocessing parameters
                const imputation = {};
                for (const col of numericalFeatures) {
                    imputation[col] = getImputationParams(combinedData, col, 'median');
                }
                
                const scalers = {};
                for (const col of numericalFeatures) {
                    scalers[col] = getScalingParams(combinedData, col);
                }
                
                const encoders = {};
                for (const col of categoricalFeatures) {
                    encoders[col] = getOneHotEncodingParams(combinedData, col);
                }
                
                const preprocessingParams = {
                    numericalFeatures,
                    categoricalFeatures,
                    targetVariable,
                    imputation,
                    scalers,
                    encoders
                };
                
                await fs.mkdir(MODEL_OUTPUT_DIR, { recursive: true });
                await fs.writeFile(paramsSavePath, JSON.stringify(preprocessingParams, null, 2));
                logger.info(`[${maintenanceType}] Preprocessing params saved to ${paramsSavePath}`);

                // --- 4. Data Transformation ---
                logger.info(`[${maintenanceType}] Applying preprocessing...`);
                const processedData = applyPreprocessing(combinedData, preprocessingParams);
                if (processedData.length === 0) {
                    throw new Error('No valid data after preprocessing');
                }

                tf.util.shuffle(processedData);
                const splitIndex = Math.floor(processedData.length * (1 - TEST_SET_RATIO));
                const trainData = processedData.slice(0, splitIndex);
                const testData = processedData.slice(splitIndex);

                // --- Create Tensors (using tf.tidy for safety) ---
                let X_train, y_train, X_test, y_test;
                tf.tidy(() => { // Use tidy to manage intermediate tensors
                    const trainFeatures = trainData.map(d => d.features);
                    const trainTargets = trainData.map(d => [d.target]);
                    const testFeatures = testData.map(d => d.features);
                    const testTargets = testData.map(d => [d.target]);

                    X_train = tf.tensor2d(trainFeatures);
                    y_train = tf.tensor2d(trainTargets);
                    X_test = tf.tensor2d(testFeatures);
                    y_test = tf.tensor2d(testTargets);

                    // Keep the final tensors outside tidy by not disposing them implicitly
                    tf.keep(X_train);
                    tf.keep(y_train);
                    tf.keep(X_test);
                    tf.keep(y_test);
                });

                // Add final tensors to disposal list
                tensorsToDispose.push(X_train, y_train, X_test, y_test);

                const inputShape = [X_train.shape[1]]; // Get shape after creation
                logger.info(`[${maintenanceType}] Data tensors created: Train ${X_train.shape[0]}, Test ${X_test.shape[0]}, Features ${inputShape[0]}`);

                // --- 5. Model Definition ---
                logger.info(`[${maintenanceType}] Creating model architecture...`);
                const model = tf.sequential();
                model.add(tf.layers.dense({ inputShape: inputShape, units: 64, activation: 'relu', kernelInitializer: 'heNormal' }));
                model.add(tf.layers.batchNormalization());
                model.add(tf.layers.dropout({ rate: 0.2 }));
                model.add(tf.layers.dense({ units: 32, activation: 'relu', kernelInitializer: 'heNormal' }));
                model.add(tf.layers.dense({ units: 1 })); // Single output for regression

                // --- 6. Model Compilation ---
                logger.info(`[${maintenanceType}] Compiling model...`);
                model.compile({
                    optimizer: tf.train.adam(0.001), // Learning rate
                    loss: 'meanSquaredError',        // MSE for regression
                    metrics: ['mse', 'mae']          // Track MSE and MAE
                });
                model.summary(); // Log summary

                // --- 7. Model Training ---
                logger.info(`[${maintenanceType}] Starting training (Epochs: 100, Batch: 8)...`);
                const history = await model.fit(X_train, y_train, {
                    epochs: 100,
                    batchSize: 8,
                    validationSplit: 0.15,
                    callbacks: [
                        tf.callbacks.earlyStopping({
                            monitor: 'val_loss',      // Monitor validation loss
                            patience: 10,             // Stop if no improvement for 10 epochs
                            minDelta: 100,            // Min change considered improvement
                            // restoreBestWeights: true, // Restore weights from best epoch
                            verbose: 1                // Log when stopping early
                        }),
                        // Custom logger callback
                        new tf.CustomCallback({
                            onEpochEnd: async (epoch, logs) => {
                                if (epoch % 10 === 0) { // Log every 10 epochs to avoid spam
                                    logger.debug(`[${maintenanceType}] Epoch ${epoch + 1} - loss: ${logs.loss.toFixed(2)}, val_loss: ${logs.val_loss.toFixed(2)}`);
                                }
                                await tf.nextFrame(); // Prevent blocking event loop on long training
                            }
                        })
                    ],
                    verbose: 0 // Set TF's internal logging level (0 = silent, 1 = progress bar, 2 = one line per epoch)
                });
                logger.info(`[${maintenanceType}] Training completed after ${history.epoch.length} epochs.`);

                // --- 8. Model Evaluation ---
                logger.info(`[${maintenanceType}] Evaluating model...`);
                const evalResult = model.evaluate(X_test, y_test); // Returns Tensor(s)
                // Ensure we get the correct tensor (often index 1 for 'mse' if metrics=['mse', 'mae'])
                const mseTensor = Array.isArray(evalResult) ? evalResult[1] : evalResult;
                tensorsToDispose.push(mseTensor); // Track for disposal

                const testMSE = (await mseTensor.data())[0]; // Get numeric value
                const testRMSE = Math.sqrt(testMSE);
                logger.info(`[${maintenanceType}] Test RMSE: ${testRMSE.toFixed(2)} engine hours`);

                // --- 9. Model Saving ---
                logger.info(`[${maintenanceType}] Saving model to ${modelSavePath}...`);
                await model.save(modelSavePath);

                results[maintenanceType] = {
                    status: 'success',
                    modelPath: modelSavePath,
                    testRMSE: testRMSE,
                    trainingSamples: trainData.length,
                    testSamples: testData.length
                };
                overallSuccess = true; // Mark that at least one model trained successfully
                logger.info(`[${maintenanceType}] Training pipeline finished successfully.`);

            } catch (error) { // Catch errors specific to this maintenance type's pipeline
                logger.error(`[${maintenanceType}] Pipeline error:`, error);
                results[maintenanceType] = {
                    status: 'failed',
                    error: error.message,
                    stack: error.stack // Include stack for debugging
                };
            } finally {
                logger.info(`--- Completed processing for ${maintenanceType} ---`);
            }
        } // End loop over maintenance types

        // --- Final Check and Tensor Cleanup ---
        logger.info('Cleaning up final tensors...');
        let disposedCount = 0;
        for (const tensor of tensorsToDispose) {
            try {
                if (tensor && !tensor.isDisposed) {
                    tensor.dispose();
                    disposedCount++;
                }
            } catch (disposeError) {
                logger.warn('Error disposing tensor:', disposeError);
            }
        }
        logger.info(`Disposed ${disposedCount}/${tensorsToDispose.length} tensors.`);

        // Check overall results after processing all types
        if (overallSuccess) {
            logger.info('Maintenance Model Training Pipeline finished (at least partially successful).');
            return results;
        } else {
            // If all models were skipped due to data issues, return results but log a warning
            const failedCount = Object.values(results).filter(r => r.status === 'failed').length;
            const skippedCount = Object.values(results).filter(r => r.status === 'skipped').length;
            
            if (failedCount > 0) {
                logger.error('One or more maintenance model trainings failed.');
            }
            
            if (skippedCount === TARGET_MAINTENANCE_TYPES.length) {
                logger.warn('All maintenance types skipped due to insufficient data. No models trained.');
            }
            
            return results;
        }

    } catch (error) { // Catch errors from the overall pipeline setup (e.g., DB connection)
        logger.error('Maintenance pipeline failed catastrophically:', error);
        // Ensure cleanup happens even on outer error
        tensorsToDispose.forEach(tensor => { 
            try { 
                if(tensor && !tensor.isDisposed) tensor.dispose(); 
            } catch(e){} 
        });
        throw error; // Re-throw for BullMQ worker
    }
};