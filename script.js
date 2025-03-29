import mongoose from 'mongoose';
import 'dotenv/config'; // Ensure environment variables like MONGO_URI are loaded
import Ship from './models/ship.model.js';
import Route from './models/route.model.js';
import Maintenance from './models/maintenance.model.js';
import FuelLog from './models/fuel.model.js';
import logger from './utils/logger.js'; // Use your logger

// --- Database Connection ---
// Ensure MONGO_URI is set in your .env file
if (!process.env.MONGO_URI) {
    logger.error('MONGO_URI environment variable is not set.');
    process.exit(1);
}

mongoose.connect(process.env.MONGO_URI, {
    // Recommended options for newer Mongoose versions
    // useNewUrlParser: true, // Deprecated
    // useUnifiedTopology: true, // Deprecated
    serverSelectionTimeoutMS: 10000, // Allow more time in dev/seeding
    socketTimeoutMS: 45000,
})
.then(() => logger.info('SEEDER: Connected to MongoDB'))
.catch(err => {
    logger.error('SEEDER: MongoDB connection error:', err);
    process.exit(1); // Exit if can't connect
});

// --- Helper Functions ---
const random = (min, max) => Math.floor(Math.random() * (max - min + 1) + min);
const randomFloat = (min, max) => parseFloat((Math.random() * (max - min) + min).toFixed(2));

// --- Sample Data Generators ---
const generateShipData = () => ({
    name: `Vessel-${random(1000, 9999)}`,
    imoNumber: `IMO${random(1000000, 9999999)}`,
    type: ['Container', 'Tanker', 'Bulk Carrier', 'Cargo'][random(0, 3)],
    buildYear: random(2000, 2023),
    capacity: { weight: random(10000, 50000), volume: random(20000, 100000) },
    engine: {
        type: ['Diesel', 'Heavy Fuel Oil', 'Marine Gas Oil'][random(0, 2)],
        power: random(10000, 30000),
        fuelType: ['MGO', 'HFO', 'VLSFO'][random(0, 2)],
    },
    dimensions: { length: random(100, 300), width: random(20, 50), draft: random(5, 15) }
});

// **MODIFIED generateRouteData**
const generateRouteData = (shipId) => {
    const statusOptions = ['Planned', 'In-Progress', 'Completed']; // Use consistent casing if needed
    const generatedStatus = statusOptions[random(0, 2)];
    const estimatedDurationHours = random(24, 720);
    const departureTime = new Date(Date.now() - random(1, 90) * 24 * 60 * 60 * 1000); // Departed up to 90 days ago

    const routeData = {
        shipId,
        departure: {
            port: ['Singapore', 'Rotterdam', 'Shanghai', 'Dubai', 'Mumbai'][random(0, 4)],
            timestamp: departureTime,
            coordinates: { lat: randomFloat(-90, 90), lng: randomFloat(-180, 180) }
        },
        destination: {
            port: ['Hong Kong', 'New York', 'Sydney', 'Cape Town', 'Santos'][random(0, 4)],
            // Estimate arrival based on estimated duration for realism
            estimatedTimestamp: new Date(departureTime.getTime() + estimatedDurationHours * 60 * 60 * 1000),
            coordinates: { lat: randomFloat(-90, 90), lng: randomFloat(-180, 180) }
        },
        distance: random(1000, 10000),
        status: generatedStatus,
        cargoWeight: random(5000, 40000),
        estimatedDuration: estimatedDurationHours,
        weatherConditions: { // Keep generating some weather info
             averageWindSpeed: random(5, 30),
             averageWaveHeight: random(1, 5),
             predominantDirection: ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'][random(0, 7)]
        },
        // **Conditionally add actualDuration only for Completed routes**
        ...(generatedStatus === 'Completed' && {
            // Make actual duration slightly different from estimated
            actualDuration: Math.max(1, Math.round(estimatedDurationHours * randomFloat(0.85, 1.15))), // Ensure > 0
        }),
    };
    return routeData;
};

const generateMaintenanceData = (shipId) => ({
    shipId,
    type: ['Engine Check', 'Hull Inspection', 'Deck Maintenance', 'Electrical Systems', 'Safety Equipment'][random(0, 4)], // Match example types?
    description: `Routine maintenance of ${['engine', 'hull', 'deck', 'electrical', 'safety'][random(0, 4)]} systems`,
    performedAt: new Date(Date.now() - random(1, 365) * 24 * 60 * 60 * 1000),
    engineHoursAtMaintenance: random(1000, 50000), // Increased range
    partsReplaced: [{
        name: ['Filter', 'Bearing', 'Pump', 'Valve', 'Seal', 'Sensor'][random(0, 5)],
        serialNumber: `SN${random(10000, 99999)}`,
        lifeExpectancy: random(1000, 5000)
    }],
    cost: random(1000, 50000),
    technician: `Tech-${random(100, 999)}`
});

// **MODIFIED generateFuelLogData** - Ensure positive quantity
const generateFuelLogData = (shipId, routeId) => ({
    shipId,
    routeId,
    timestamp: new Date(Date.now() - random(0, 30) * 24 * 60 * 60 * 1000), // Logged sometime in last 30 days
    fuelType: ['MGO', 'HFO', 'VLSFO'][random(0, 2)],
    quantity: random(500, 5000), // Ensure positive quantity > 0
    consumptionRate: randomFloat(50, 200),
    engineHours: random(1000, 50000), // Increased range, should correlate with maintenance eventually
    rpm: random(80, 120),
    speed: random(10, 25),
    weatherConditions: { windSpeed: random(5, 30), waveHeight: random(1, 5), temperature: random(15, 35) },
    notes: 'Generated fuel consumption log'
});

// --- Main Seeding Function ---
async function seedData() {
    try {
        logger.info('SEEDER: Starting data seeding process...');
        // Clear existing data
        logger.info('SEEDER: Clearing existing data...');
        await Promise.all([
            Ship.deleteMany({}),
            Route.deleteMany({}),
            Maintenance.deleteMany({}),
            FuelLog.deleteMany({})
        ]);
        logger.info('SEEDER: Existing data cleared.');

        // Create ships
        logger.info('SEEDER: Creating ships...');
        const ships = await Ship.insertMany(
            Array(20).fill().map(() => generateShipData()) // Create 20 ships
        );
        logger.info(`SEEDER: Ships created: ${ships.length}`);

        // Create routes for each ship (Ensure some are Completed with actualDuration)
        logger.info('SEEDER: Creating routes...');
        const allRoutes = [];
        for (const ship of ships) {
            const shipRoutes = await Route.insertMany(
                Array(random(5, 10)).fill().map(() => generateRouteData(ship._id)) // Create 5-10 routes per ship
            );
            allRoutes.push(...shipRoutes);
        }
        logger.info(`SEEDER: Routes created: ${allRoutes.length}`);

        // Filter for routes that are suitable for linking fuel logs (Completed or In-Progress)
        // Analytics only uses 'Completed', but linking logs to 'In-Progress' is also realistic
        const routesForFuelLogs = allRoutes.filter(r => r.status === 'Completed' || r.status === 'In-Progress');
        logger.info(`SEEDER: Routes suitable for fuel logs: ${routesForFuelLogs.length}`);

        // Create maintenance records (1 per ship for simplicity)
        logger.info('SEEDER: Creating maintenance records...');
        const maintenance = await Maintenance.insertMany(
            ships.map(ship => generateMaintenanceData(ship._id))
        );
        logger.info(`SEEDER: Maintenance records created: ${maintenance.length}`);

        // Create fuel logs (Multiple logs per VALID route)
        logger.info('SEEDER: Creating fuel logs...');
        const fuelLogsData = [];
        for (const route of routesForFuelLogs) {
            const numLogs = random(3, 8); // Add 3-8 logs per suitable route
            for (let i = 0; i < numLogs; i++) {
                fuelLogsData.push(generateFuelLogData(route.shipId, route._id));
            }
        }
        if (fuelLogsData.length > 0) {
             const fuelLogs = await FuelLog.insertMany(fuelLogsData);
             logger.info(`SEEDER: Fuel logs created: ${fuelLogs.length}`);
        } else {
             logger.warn('SEEDER: No suitable routes found to create fuel logs.');
        }


        logger.info('SEEDER: Data seeding completed successfully!');

    } catch (error) {
        logger.error('SEEDER: Error seeding data:', error);
        process.exitCode = 1; // Indicate error
    } finally {
        // Disconnect mongoose connection
        logger.info('SEEDER: Disconnecting from MongoDB...');
        await mongoose.disconnect();
        logger.info('SEEDER: Disconnected.');
        process.exit(process.exitCode || 0); // Exit with appropriate code
    }
}

// --- Run the Seeding ---
seedData();