// internal/people/PeopleRepository.js
const { DBConnector } = require('../../modules/DBConnector');

class PeopleRepository {
    static db = new DBConnector('people.json');

    static read() {
        try {
            const file = this.db.readFile();
            return file.trim() ? JSON.parse(file) : [];
        } catch (err) {
            if (err.code === 'ENOENT') return [];
            throw err;
        }
    }

    static write(data) {
        this.db.writeFile(JSON.stringify(data, null, 2));
    }
}

module.exports = { PeopleRepository };