import { Router } from 'express';
import { 
    addMaintenance,
    getMaintenanceSchedule
} from '../controllers/maintenance.controller.js';

const router = Router();

router.post('/add-maintenance', addMaintenance);
router.get('/schedule', getMaintenanceSchedule);

export default router;