// API/internal/people/PeopleService.js
const { PersonDAO } = require('./PerosnDAO');

class PeopleService {
    static getAll() {
        return PersonDAO.findAll().map(p => p.toJSON());
    }

    static getById(id) {
        return PersonDAO.findById(id).toJSON();
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