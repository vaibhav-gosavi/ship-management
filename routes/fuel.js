import { Router } from 'express';
import { 
    addFuelLog,
    getFuelEstimate 
} from '../controllers/fuel.controller.js';

const router = Router();

router.post('/logs', addFuelLog);
router.get('/estimate', getFuelEstimate);

export default router;