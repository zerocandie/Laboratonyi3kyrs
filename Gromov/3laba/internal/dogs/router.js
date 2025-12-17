const express = require('express');
const router = express.Router();
const { DogsController } = require('./DogsController');

router.get('/', DogsController.getAll);
router.get('/:id', DogsController.getById);
router.post('/', DogsController.create);
router.delete('/:id', DogsController.remove);

module.exports = router;