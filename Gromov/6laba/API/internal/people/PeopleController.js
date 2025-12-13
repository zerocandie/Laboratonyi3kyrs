// API/internal/people/PeopleController.js
const { PeopleService } = require('./PeopleService.js');

class PeopleController {
    static getAll(req, res) {
        try {
            res.json(PeopleService.getAll());
        } catch (err) {
            res.status(500).json({ error: err.message });
        }
    }

    static getById(req, res) {
        try {
            const { id } = req.params;
            res.json(PeopleService.getById(id));
        } catch (err) {
            res.status(404).json({ error: err.message });
        }
    }

    static create(req, res) {
        try {
            const person = PeopleService.create(req.body);
            res.status(201).json(person);
        } catch (err) {
            res.status(400).json({ error: err.message });
        }
    }

    static remove(req, res) {
        try {
            const { id } = req.params;
            PeopleService.remove(id);
            res.json({ message: `Человек ${id} удалён` });
        } catch (err) {
            res.status(404).json({ error: err.message });
        }
    }
}

module.exports = { PeopleController };