const express = require('express')
const router = express.Router();

const { PeopleController } = require('./PeopleController');

router.get('/', PeopleController.getAll);
router.get('/:id', PeopleController.getById);
router.post('/', PeopleController.create);
router.delete('/:id', PeopleController.remove);

module.exports = router;