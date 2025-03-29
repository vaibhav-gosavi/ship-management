import { Router } from 'express';
import { 
    addRoute, 
    addShip,
    completeRoute,
} from '../controllers/route.controller.js';

const router = Router();

router.post('/addroute', addRoute);
router.post('/ship', addShip);
router.put('/:id/complete', completeRoute);


export default router;