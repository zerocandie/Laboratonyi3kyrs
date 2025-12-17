// API/internal/people/router.js
const express = require('express');
const router = express.Router();

const { PeopleService } = require('./PeopleService');

// GET /api/people
router.get('/', (req, res) => {
  res.json(PeopleService.getAll());
});

// GET /api/people/:id
router.get('/:id', (req, res) => {
  const person = PeopleService.getById(req.params.id);
  if (person) {
    res.json(person);
  } else {
    res.status(404).json({ error: 'Не найдено' });
  }
});

// DELETE /api/people/:id
router.delete('/:id', (req, res) => {
  const deleted = PeopleService.deleteById(req.params.id);
  if (deleted) {
    res.status(204).end();
  } else {
    res.status(404).json({ error: 'Не найдено' });
  }
});

module.exports = router;