// API/internal/people/PersonDAO.js
const { PeopleRepository } = require('./peopleRepository');

class PersonDAO {
    constructor(id, name, role, photo, description) {
        this.id = id;
        this.name = name;
        this.role = role;
        this.photo = photo;
        this.description = description;
    }

    static findAll() {
        const people = PeopleRepository.read();
        return people.map(p => new this(p.id, p.name, p.role, p.photo, p.description));
    }

    static findById(id) {
        const people = PeopleRepository.read();
        const p = people.find(x => x.id == id);
        if (!p) throw new Error('Человек не найден');
        return new this(p.id, p.name, p.role, p.photo, p.description);
    }

    static create(person) {
    
    const people = PeopleRepository.read();
    if (people.some(p => p.id === person.id)) {
        throw new Error('ID уже существует');
    }
    PeopleRepository.write([...people, person]);
    return new this(person.id, person.name, person.role, person.photo, person.description);
}

    static delete(id) {
        const people = PeopleRepository.read();
        const filtered = people.filter(p => p.id != id);
        if (filtered.length === people.length) throw new Error('Не найден');
        PeopleRepository.write(filtered);
        return true;
    }

    toJSON() {
        return { id: this.id, name: this.name, role: this.role, photo: this.photo, description: this.description };
    }
}

module.exports = { PersonDAO };