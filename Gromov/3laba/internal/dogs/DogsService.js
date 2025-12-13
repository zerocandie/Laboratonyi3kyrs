const { DogDAO } = require('./DogDAO');

class DogsService {
    static getAll() {
        return DogDAO.findAll().map(d => d.toJSON());
    }

    static getById(id) {
        return DogDAO.findById(id).toJSON();
    }

    static create(dog) {
        return DogDAO.create(dog).toJSON();
    }

    static remove(id) {
        DogDAO.delete(id);
        return { message: `Порода "${id}" удалена` };
    }
}

module.exports = { DogsService };