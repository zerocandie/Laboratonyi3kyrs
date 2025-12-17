// API/modules/DBConnector.js
const fs = require('fs');
const path = require('path');

class DBConnector {
    constructor(filename) {
        // Сохраняем полный путь к файлу в КОРНЕ проекта
        this.filePath = path.join(__dirname, '..', '..', filename);
    }

    readFile() {
        return fs.readFileSync(this.filePath, 'utf8');
    }

    writeFile(data) {
        
        fs.writeFileSync(this.filePath, data, 'utf8');
    }
}

module.exports = { DBConnector };