const { DBConnector } = require('../../modules/DBConnector');

class DogsRepository {
    static db = new DBConnector('dogs.json'); // ← отдельный файл!

    static read() {
        try {
            const file = this.db.readFile();
            return file.trim() ? JSON.parse(file) : [];
        } catch (err) {
            if (err.code === 'ENOENT') return [];
            throw new Error('Ошибка чтения dogs.json: ' + err.message);
        }
    }

    static write(data) {
        this.db.writeFile(JSON.stringify(data, null, 2));
    }
}

module.exports = { DogsRepository };