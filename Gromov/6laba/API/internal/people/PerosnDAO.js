// API/internal/people/PersonDAO.js
const fs = require('fs');
const path = require('path');

// Путь к data.json в корне проекта
const DB_PATH = path.join(__dirname, '..', '..', '..', 'data.json');

function readDB() {
    const rawData = fs.readFileSync(DB_PATH, 'utf8');
    return JSON.parse(rawData);
}

function writeDB(data) {
    fs.writeFileSync(DB_PATH, JSON.stringify(data, null, 2), 'utf8');
}

class PersonDAO {
    static findAll() {
        const db = readDB();
        return db.people || [];
    }

    static findById(id) {
        const people = this.findAll();
        return people.find(p => p.id === id) || null;
    }

    static create(person) {
        const db = readDB();
        db.people = db.people || [];
        db.people.push(person);
        writeDB(db);
        return person;
    }

    static delete(id) {
        const db = readDB();
        db.people = (db.people || []).filter(p => p.id !== id);
        writeDB(db);
    }
}

module.exports = { PersonDAO };