const { DogsService } = require('./DogsService');

class DogsController {
    static getAll(req, res) {
        try {
            res.json(DogsService.getAll());
        } catch (err) {
            res.status(500).json({ error: err.message });
        }
    }

    static getById(req, res) {
        try {
            const { id } = req.params;
            res.json(DogsService.getById(id));
        } catch (err) {
            res.status(404).json({ error: err.message });
        }
    }

    static create(req, res) {
        try {
            const dog = DogsService.create(req.body);
            res.status(201).json(dog);
        } catch (err) {
            res.status(400).json({ error: err.message });
        }
    }

    static remove(req, res) {
        try {
            const { id } = req.params;
            const result = DogsService.remove(id);
            res.json(result);
        } catch (err) {
            res.status(404).json({ error: err.message });
        }
    }
}

module.exports = { DogsController };