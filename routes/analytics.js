import { Router } from 'express';
import { getAnalytics } from '../controllers/analytics.controller.js';

const router = Router();

router.get('/get-analytics', getAnalytics);

export default router;