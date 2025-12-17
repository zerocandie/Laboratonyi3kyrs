// API/internal/people/PeopleService.js
const { PersonDAO } = require('./PerosnDAO');

class PeopleService {
    static getAll() {
        return PersonDAO.findAll().map(p => p.toJSON());
    }

    static getById(id) {
        const person = PersonDAO.findById(id);
        return person ? person.toJSON() : null;
    }

    static create(person) {
        return PersonDAO.create(person).toJSON();
    }

    static remove(id) {
        PersonDAO.delete(id);
        return true;
    }
}

module.exports = { PeopleService };