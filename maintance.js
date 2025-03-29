// scripts/seedMaintenance.js
import mongoose from 'mongoose';
import 'dotenv/config'; // Ensure environment variables are loaded
import { faker } from '@faker-js/faker'; // For generating realistic fake data
import Maintenance from './models/maintenance.model.js'; // Adjust path if needed
// Optional: Import Ship model if you want to base data on ship type etc.
// import Ship from '../models/ship.model.js';
import logger from './utils/logger.js'; // Use your logger

// --- Configuration ---
const DELETE_EXISTING_DATA = true; // !!! SET TO true TO DELETE BEFORE SEEDING !!!
const NUMBER_OF_RECORDS_PER_TYPE_PER_SHIP = 5; // How many historical records to create
const MAINTENANCE_TYPES = ['Engine Overhaul', 'Hull Inspection', 'Electrical Systems Check', 'Propeller Maintenance'];
const TECHNICIANS = ['John Smith', 'Maria Garcia', 'Ahmed Khan', 'Chen Wei', 'Olga Petrova'];

// --- !!! PASTE YOUR ACTUAL SHIP IDs HERE !!! ---
// Ensure these IDs exist in your 'ships' collection
const SHIP_IDS = [
    '67e7a8c08f1e26525b3a6be3', // Replace with YOUR valid ship IDs
    '67e7a8c08f1e26525b3a6be4',
    '67e7a8c08f1e26525b3a6be5',
    '67e7a8c08f1e26525b3a6be6',
    '67e7a8c08f1e26525b3a6be7',
    '67e7a8c08f1e26525b3a6be8',
    '67e7a8c08f1e26525b3a6be9',
    '67e7a8c08f1e26525b3a6bea',
    '67e7a8c08f1e26525b3a6beb',
    '67e7a8c08f1e26525b3a6bec',
    '67e7a8c08f1e26525b3a6bed',
    '67e7a8c08f1e26525b3a6bee',
    '67e7a8c08f1e26525b3a6bef',
    '67e7a8c08f1e26525b3a6bf0',
    '67e7a8c08f1e26525b3a6bf1',
    '67e7a8c08f1e26525b3a6bf2',
    '67e7a8c08f1e26525b3a6bf3',
    // Add more valid ship IDs from your database
];
// ----------------------------------------------

// --- Helper Functions ---
function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function addMonths(date, months) {
    const d = new Date(date);
    d.setMonth(d.getMonth() + months);
    // Handle day overflow (e.g., Jan 31 + 1 month = Feb 28/29)
    if (d.getDate() < date.getDate()) {
        d.setDate(0); // Go to last day of previous month
    }
    // Add some random days for variability
    d.setDate(d.getDate() + getRandomInt(-5, 5));
    return d;
}

// --- Main Seeding Function ---
const seedMaintenanceData = async () => {
    let totalCreated = 0;
    try {
        logger.info('Connecting to MongoDB...');
        await mongoose.connect(process.env.MONGO_URI);
        logger.info('MongoDB connected successfully.');

        if (SHIP_IDS.length === 0) {
             logger.error('No SHIP_IDS provided in the script. Exiting.');
             return;
        }

        if (DELETE_EXISTING_DATA) {
            logger.warn('!!! DELETING existing maintenance data !!!');
            const { deletedCount } = await Maintenance.deleteMany({}); // Delete ALL maintenance records
            logger.info(`Deleted ${deletedCount} existing maintenance records.`);
        } else {
            logger.info('Skipping deletion of existing maintenance data.');
        }

        logger.info(`Generating maintenance data for ${SHIP_IDS.length} ships...`);
        const allMaintenanceRecords = [];

        for (const shipId of SHIP_IDS) {
            if (!mongoose.Types.ObjectId.isValid(shipId)) {
                logger.warn(`Skipping invalid Ship ID format: ${shipId}`);
                continue;
            }
            logger.debug(`Processing Ship ID: ${shipId}`);

            for (const type of MAINTENANCE_TYPES) {
                logger.debug(` -> Generating type: ${type}`);
                let lastPerformedAt = faker.date.past({ years: 3 }); // Start 3 years ago
                // Base engine hours - give some variability per ship/type
                let lastEngineHours = getRandomInt(1000, 5000);

                for (let i = 0; i < NUMBER_OF_RECORDS_PER_TYPE_PER_SHIP; i++) {
                    // 1. Calculate next date (add 4-12 months)
                    const nextPerformedAt = addMonths(lastPerformedAt, getRandomInt(4, 12));

                    // 2. Calculate next engine hours (add 500-3000 hours - MUST BE POSITIVE)
                    const engineHoursIncrement = getRandomInt(500, 3000);
                    const nextEngineHours = lastEngineHours + engineHoursIncrement;

                    // 3. Create the record object
                    const maintenanceRecord = {
                        shipId: new mongoose.Types.ObjectId(shipId), // Ensure it's an ObjectId
                        type: type,
                        description: `Performed ${type.toLowerCase()} - ${faker.lorem.sentence({ min: 3, max: 8 })}`,
                        performedAt: nextPerformedAt,
                        engineHoursAtMaintenance: nextEngineHours,
                        cost: parseFloat(faker.commerce.price({ min: 500, max: 15000, dec: 2 })),
                        technician: faker.helpers.arrayElement(TECHNICIANS),
                        // Optional: Add some parts replaced sometimes
                        partsReplaced: Math.random() > 0.6 ? // ~40% chance of having parts
                            [...Array(getRandomInt(1, 3))].map(() => ({ // 1 to 3 parts
                                name: faker.commerce.productName(),
                                serialNumber: faker.string.alphanumeric(12).toUpperCase(),
                                lifeExpectancy: getRandomInt(1000, 5000) // hours
                            }))
                            : [],
                        // Optional: Set nextDue based on performedAt (e.g., 6 months later)
                        // nextDue: addMonths(nextPerformedAt, 6),
                        createdAt: new Date(), // Set creation/update times
                        updatedAt: new Date()
                    };

                    allMaintenanceRecords.push(maintenanceRecord);

                    // Update last values for the next iteration for *this ship and type*
                    lastPerformedAt = nextPerformedAt;
                    lastEngineHours = nextEngineHours;
                } // end loop for number of records
            } // end loop for maintenance types
        } // end loop for ship IDs

        // --- Insert all generated records ---
        if (allMaintenanceRecords.length > 0) {
            logger.info(`Attempting to insert ${allMaintenanceRecords.length} new maintenance records...`);
            const inserted = await Maintenance.insertMany(allMaintenanceRecords, { ordered: false }); // ordered: false can help if some docs fail validation
            totalCreated = inserted.length;
            logger.info(`Successfully inserted ${totalCreated} maintenance records.`);
        } else {
            logger.warn('No maintenance records were generated to insert.');
        }

    } catch (error) {
        logger.error('Error during maintenance data seeding:', error);
    } finally {
        logger.info('Disconnecting from MongoDB...');
        await mongoose.disconnect();
        logger.info('MongoDB disconnected.');
        logger.info(`--- Seeding Complete --- Total Records Created: ${totalCreated} ---`);
    }
};

// --- Run the Seeding Function ---
seedMaintenanceData();