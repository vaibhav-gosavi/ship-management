import { Router } from 'express';
import { 
    triggerTraining
} from '../controllers/admin.controller.js';

const router = Router();

router.post('/trigger-training', triggerTraining);


export default router;