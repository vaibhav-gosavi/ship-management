import { Router } from 'express';
// import shipRoutes from './ships.js';
import routeRoutes from './routes.js';
import fuelRoutes from './fuel.js';
import maintenanceRoutes from './maintenance.js';
import adminRoutes from './admin.js';
import analyticsRoutes from './analytics.js';

const router = Router();

// Mount all routes under /api/v1
// router.use('/api/v1/ships', shipRoutes);
router.use('/api/v1/routes', routeRoutes);
router.use('/api/v1/fuel', fuelRoutes);
router.use('/api/v1/maintenance', maintenanceRoutes);
router.use('/api/v1/analytics', analyticsRoutes);
router.use('/api/v1/admin', adminRoutes);

export default router;