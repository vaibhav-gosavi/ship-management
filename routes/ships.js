import { Router } from 'express';
import { 
    addShip,
    listShips 
} from '../controllers/route.controller.js';

const router = Router();

router.post('/', addShip);
router.get('/', listShips);

export default router;