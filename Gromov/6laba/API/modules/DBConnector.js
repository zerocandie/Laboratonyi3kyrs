const fs = require('fs');
const path = require('path');

class DBConnector {
    constructor(filename) {
        this.filename = filename;
    }

    readFile() {
        const fullPath = path.join(process.cwd(), 'db', this.filename);
        return fs.readFileSync(fullPath, 'utf8');
    }

    writeFile(data) {
        const fullPath = path.join(process.cwd(), 'db', this.filename);
        fs.mkdirSync(path.dirname(fullPath), { recursive: true });
        fs.writeFileSync(fullPath, data, 'utf8');
    }
}

module.exports = { DBConnector };