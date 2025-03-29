import mongoose from 'mongoose';
import Route from '../models/route.model.js';
import FuelLog from '../models/fuel.model.js'; // Needed if fuel isn't stored on Route
import logger from '../utils/logger.js';

/**
 * @desc    Get analytics insights on route efficiency and fuel trends
 * @route   GET /api/analytics
 * @access  Private (Add auth middleware)
 */
export const getAnalytics = async (req, res, next) => {
    try {
        logger.info('Fetching analytics data...');

        // --- Aggregation Pipeline ---
        const analyticsPipeline = [
            // 1. Filter for completed routes with necessary data
            {
                $match: {
                    status: 'Completed',
                    actualDuration: { $exists: true, $ne: null },
                    distance: { $exists: true, $ne: null, $gt: 0 }, // Ensure distance > 0
                    estimatedDuration: { $exists: true, $ne: null }
                    // Optional: Add date range filters based on req.query (e.g., ?startDate=...&endDate=...)
                    // 'departure.timestamp': { $gte: startDate, $lte: endDate }
                }
            },
            // 2. Lookup Ship Information
            {
                $lookup: {
                    from: 'ships', // The actual collection name for Ships
                    localField: 'shipId',
                    foreignField: '_id',
                    as: 'shipInfo'
                }
            },
            // Only proceed if shipInfo is found
            { $match: { 'shipInfo.0': { $exists: true } } },
            { $addFields: { shipInfo: { $first: '$shipInfo' } } }, // Deconstruct shipInfo array

            // 3. Lookup and Sum Fuel Logs for each Route (More robust if totalFuel was stored on Route)
            {
                $lookup: {
                    from: 'fuellogs', // The actual collection name for FuelLogs
                    localField: '_id',
                    foreignField: 'routeId',
                    pipeline: [ // Calculate sum within the lookup
                        { $match: { quantity: { $exists: true, $ne: null, $gt: 0 } } },
                        { $group: { _id: null, totalFuel: { $sum: '$quantity' } } }
                    ],
                    as: 'fuelData'
                }
            },
            // Only proceed if fuelData is found and has totalFuel > 0
             { $match: { 'fuelData.0': { $exists: true } } },
             { $addFields: { totalFuelConsumed: { $ifNull: [ { $first: '$fuelData.totalFuel' }, 0 ] } } },
             { $match: { totalFuelConsumed: { $gt: 0 } } }, // Filter out routes with zero fuel

            // 4. Project calculated fields per route
            {
                $project: {
                    _id: 1, // Keep route ID if needed later
                    shipId: 1,
                    shipType: '$shipInfo.type',
                    shipName: '$shipInfo.name',
                    distance: 1,
                    cargoWeight: 1,
                    actualDuration: 1,
                    estimatedDuration: 1,
                    totalFuelConsumed: 1,
                    departureTimestamp: '$departure.timestamp',
                    // Calculate metrics
                    fuelPerNM: { $cond: [ { $gt: ['$distance', 0] }, { $divide: ['$totalFuelConsumed', '$distance'] }, null ] },
                    durationAccuracyRatio: { $cond: [ { $gt: ['$estimatedDuration', 0] }, { $divide: ['$actualDuration', '$estimatedDuration'] }, null ] }, // Actual / Estimated
                    avgSpeed: { $cond: [ { $gt: ['$actualDuration', 0] }, { $divide: ['$distance', '$actualDuration'] }, null ] } // Knots (if distance is NM, duration is hours)
                }
            },
            // 5. Group to calculate overall and grouped statistics (using $facet for parallel aggregations)
            {
                $facet: {
                    // Overall Statistics
                    "overallStats": [
                        {
                            $group: {
                                _id: null,
                                totalCompletedRoutes: { $count: {} },
                                totalDistanceTravelled: { $sum: '$distance' },
                                totalFuelConsumed: { $sum: '$totalFuelConsumed' },
                                avgFuelPerNM: { $avg: '$fuelPerNM' },
                                avgDurationAccuracyRatio: { $avg: '$durationAccuracyRatio' },
                                avgSpeedOverall: { $avg: '$avgSpeed' }
                            }
                        }
                    ],
                    // Stats per Ship Type
                    "statsByShipType": [
                        {
                            $group: {
                                _id: '$shipType', // Group by ship type
                                count: { $count: {} },
                                avgFuelPerNM: { $avg: '$fuelPerNM' },
                                avgDurationAccuracyRatio: { $avg: '$durationAccuracyRatio' },
                                totalFuel: { $sum: '$totalFuelConsumed'}
                            }
                        },
                        { $sort: { count: -1 } } // Sort by most frequent type
                    ],
                    // Trend Data (e.g., Monthly Fuel Efficiency)
                     "monthlyTrends": [
                         {
                            $group: {
                                _id: { // Group by year and month
                                    year: { $year: '$departureTimestamp' },
                                    month: { $month: '$departureTimestamp' }
                                },
                                avgFuelPerNM: { $avg: '$fuelPerNM' },
                                totalFuel: { $sum: '$totalFuelConsumed' },
                                routeCount: { $count: {} }
                            }
                        },
                        { $sort: { '_id.year': 1, '_id.month': 1 } }, // Sort chronologically
                        {
                            $project: { // Prettier output
                                _id: 0,
                                year: '$_id.year',
                                month: '$_id.month',
                                avgFuelPerNM: 1,
                                totalFuel: 1,
                                routeCount: 1
                            }
                        }
                     ],
                     // Top 5 Most Efficient Routes (Lowest Fuel/NM)
                    "mostEfficientRoutes": [
                        { $match: { fuelPerNM: { $ne: null } } },
                        { $sort: { fuelPerNM: 1 } }, // Ascending order
                        { $limit: 5 },
                        { $project: { shipName: 1, shipType: 1, distance: 1, totalFuelConsumed: 1, fuelPerNM: 1 } } // Select relevant fields
                    ],
                    // Top 5 Least Efficient Routes (Highest Fuel/NM)
                    "leastEfficientRoutes": [
                        { $match: { fuelPerNM: { $ne: null } } },
                        { $sort: { fuelPerNM: -1 } }, // Descending order
                        { $limit: 5 },
                        { $project: { shipName: 1, shipType: 1, distance: 1, totalFuelConsumed: 1, fuelPerNM: 1 } }
                    ]
                }
            }
        ];

        const analyticsResult = await Route.aggregate(analyticsPipeline);

        // The result is an array with one element containing the facets
        if (!analyticsResult || analyticsResult.length === 0) {
            logger.warn('No analytics data generated. No completed routes found matching criteria.');
            return res.status(404).json({ message: 'No analytics data available.' });
        }

        // Structure the response
        const responseData = {
            overallStats: analyticsResult[0].overallStats[0] || {}, // Get first element or empty object
            statsByShipType: analyticsResult[0].statsByShipType || [],
            monthlyTrends: analyticsResult[0].monthlyTrends || [],
            mostEfficientRoutes: analyticsResult[0].mostEfficientRoutes || [],
            leastEfficientRoutes: analyticsResult[0].leastEfficientRoutes || [],
            generatedAt: new Date().toISOString()
        };

        logger.info('Analytics data fetched successfully.');
        res.status(200).json(responseData);

    } catch (error) {
        logger.error('Error fetching analytics data:', error);
        next(error); // Pass to global error handler
    }
};